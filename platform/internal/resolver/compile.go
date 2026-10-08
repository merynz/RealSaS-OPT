package resolver

import (
	"context"
	"fmt"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

const (
	StageResultArtifactType = "RealSaS.StageResultManifest"
	StageResultSchema       = "v1"
)

type Action string

const (
	Reuse   Action = "REUSE"
	Execute Action = "EXECUTE"
)

type Catalog interface {
	FindQualified(ctx context.Context, artifactType, schemaVersion, semanticSHA string) (uuid.UUID, bool, error)
}

type ResolvedStage struct {
	StageID                string
	Action                 Action
	ExpectedSemanticSHA256 string
	ReusableArtifactID     *uuid.UUID
	Reason                 string
}

type Plan struct {
	TargetStageID string
	Stages        []ResolvedStage
}

func (p Plan) ExecuteStageIDs() []string {
	out := make([]string, 0)
	for _, row := range p.Stages {
		if row.Action == Execute {
			out = append(out, row.StageID)
		}
	}
	return out
}

func (p Plan) ReusedStageIDs() []string {
	out := make([]string, 0)
	for _, row := range p.Stages {
		if row.Action == Reuse {
			out = append(out, row.StageID)
		}
	}
	return out
}

func Resolve(
	ctx context.Context,
	g *stagegraph.Graph,
	catalog Catalog,
	targetStageID string,
	rootInputs []domain.ArtifactInputIdentity,
	versions map[string]release.StageVersion,
) (Plan, error) {
	if len(rootInputs) == 0 {
		return Plan{}, fmt.Errorf("compile resolver requires immutable subject root inputs")
	}
	required, err := g.AncestorsIncluding(targetStageID)
	if err != nil {
		return Plan{}, err
	}
	expected := make(map[string]string, len(required))
	out := Plan{TargetStageID: targetStageID, Stages: make([]ResolvedStage, 0, len(required))}

	for _, stageID := range required {
		stage, ok := g.Get(stageID)
		if !ok {
			return Plan{}, fmt.Errorf("unknown stage %s", stageID)
		}
		version, ok := versions[stageID]
		if !ok {
			return Plan{}, fmt.Errorf("missing stage version %s", stageID)
		}

		inputs := make([]domain.ArtifactInputIdentity, 0, len(rootInputs)+len(stage.DependsOn))
		for _, item := range rootInputs {
			if !stage.ConsumesInput(item.Role) {
				continue
			}
			inputs = append(inputs, domain.ArtifactInputIdentity{
				Role:           item.Role,
				Ordinal:        len(inputs),
				ArtifactType:   item.ArtifactType,
				SemanticSHA256: item.SemanticSHA256,
			})
		}
		for _, parent := range stage.DependsOn {
			parentSHA, ok := expected[parent]
			if !ok {
				return Plan{}, fmt.Errorf("resolver dependency order drift: %s <- %s", stageID, parent)
			}
			inputs = append(inputs, domain.ArtifactInputIdentity{
				Role:           "stage:" + parent,
				Ordinal:        len(inputs),
				ArtifactType:   StageResultArtifactType,
				SemanticSHA256: parentSHA,
			})
		}

		descriptor := domain.ArtifactSemanticDescriptor{
			ArtifactType:         StageResultArtifactType,
			SchemaVersion:        StageResultSchema,
			ProducerContract:     stageID,
			ImplementationSHA256: version.ImplementationSHA256,
			PolicySHA256:         version.PolicySHA256,
			Inputs:               inputs,
			SemanticParameters: map[string]any{
				"stage_id":          stageID,
				"parameters_sha256": version.ParametersSHA256,
			},
		}
		semanticSHA, err := descriptor.SemanticSHA256()
		if err != nil {
			return Plan{}, err
		}
		expected[stageID] = semanticSHA

		artifactID, hit, err := catalog.FindQualified(ctx, StageResultArtifactType, StageResultSchema, semanticSHA)
		if err != nil {
			return Plan{}, err
		}
		row := ResolvedStage{
			StageID:                stageID,
			Action:                 Execute,
			ExpectedSemanticSHA256: semanticSHA,
			Reason:                 "QUALIFIED_SEMANTIC_IDENTITY_MISS",
		}
		if hit {
			id := artifactID
			row.Action = Reuse
			row.ReusableArtifactID = &id
			row.Reason = "QUALIFIED_SEMANTIC_IDENTITY_HIT"
		}
		out.Stages = append(out.Stages, row)
	}
	return out, nil
}
