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
	return ResolveWithStageSemanticParameters(
		ctx,
		g,
		catalog,
		targetStageID,
		rootInputs,
		versions,
		nil,
	)
}

// ResolveWithStageSemanticParameters is the repair-safe resolver entrypoint.
//
// extraByStage is semantic input, not execution metadata. A value injected at
// one restart stage changes that stage's semantic identity; ordinary dependency
// hashing then invalidates only its downstream descendants while exact upstream
// qualified artifacts remain reusable.
//
// Reserved semantic fields are owned by the resolver and cannot be overridden.
func ResolveWithStageSemanticParameters(
	ctx context.Context,
	g *stagegraph.Graph,
	catalog Catalog,
	targetStageID string,
	rootInputs []domain.ArtifactInputIdentity,
	versions map[string]release.StageVersion,
	extraByStage map[string]map[string]any,
) (Plan, error) {
	if len(rootInputs) == 0 {
		return Plan{}, fmt.Errorf("compile resolver requires immutable subject root inputs")
	}
	required, err := g.AncestorsIncluding(targetStageID)
	if err != nil {
		return Plan{}, err
	}
	requiredSet := make(map[string]struct{}, len(required))
	for _, stageID := range required {
		requiredSet[stageID] = struct{}{}
	}
	for stageID, params := range extraByStage {
		if _, ok := g.Get(stageID); !ok {
			return Plan{}, fmt.Errorf("semantic parameter override references unknown stage %s", stageID)
		}
		if _, ok := requiredSet[stageID]; !ok {
			return Plan{}, fmt.Errorf(
				"semantic parameter override stage %s is outside target closure %s",
				stageID,
				targetStageID,
			)
		}
		for key := range params {
			if key == "stage_id" || key == "parameters_sha256" {
				return Plan{}, fmt.Errorf(
					"semantic parameter override uses reserved key %s at stage %s",
					key,
					stageID,
				)
			}
		}
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

		semanticParameters := map[string]any{
			"stage_id":          stageID,
			"parameters_sha256": version.ParametersSHA256,
		}
		for key, value := range extraByStage[stageID] {
			semanticParameters[key] = value
		}
		descriptor := domain.ArtifactSemanticDescriptor{
			ArtifactType:         StageResultArtifactType,
			SchemaVersion:        StageResultSchema,
			ProducerContract:     stageID,
			ImplementationSHA256: version.ImplementationSHA256,
			PolicySHA256:         version.PolicySHA256,
			Inputs:               inputs,
			SemanticParameters:   semanticParameters,
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
