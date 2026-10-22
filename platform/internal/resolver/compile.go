package resolver

import (
	"context"
	"fmt"
	"sort"
	"strings"

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
	return ResolveWithExternalAndModels(ctx, g, catalog, targetStageID, rootInputs, versions, nil, nil)
}

// ResolveWithExternal adds immutable, stage-scoped raw external evidence to the
// semantic descriptor of exactly the consuming stage. Descendants inherit the
// change through the parent stage semantic hash. External inputs do not create
// qualification; the consuming stage must still execute and PASS.
func ResolveWithExternal(
	ctx context.Context,
	g *stagegraph.Graph,
	catalog Catalog,
	targetStageID string,
	rootInputs []domain.ArtifactInputIdentity,
	versions map[string]release.StageVersion,
	external map[string][]domain.ArtifactInputIdentity,
) (Plan, error) {
	return ResolveWithExternalAndModels(ctx, g, catalog, targetStageID, rootInputs, versions, external, nil)
}

func normalizeModelInputs(models []domain.ArtifactInputIdentity) ([]domain.ArtifactInputIdentity, error) {
	ordered := append([]domain.ArtifactInputIdentity(nil), models...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Role < ordered[j].Role })
	prior := ""
	for index := range ordered {
		row := &ordered[index]
		if !strings.HasPrefix(row.Role, "model:") || strings.TrimPrefix(row.Role, "model:") == "" || row.ArtifactType == "" || row.SemanticSHA256 == "" {
			return nil, fmt.Errorf("model semantic input identity incomplete: %s", row.Role)
		}
		if index > 0 && row.Role == prior {
			return nil, fmt.Errorf("duplicate model semantic input: %s", row.Role)
		}
		row.Ordinal = index
		prior = row.Role
	}
	return ordered, nil
}

// ResolveWithExternalAndModels binds the immutable model snapshot selected at
// compile acceptance into stage semantic identity. For the current fit-era DAG
// every stage is conservatively model-bound. A future inference-only release may
// narrow this to explicitly declared model-consuming stages without changing the
// exact ModelRelease identity contract.
func ResolveWithExternalAndModels(
	ctx context.Context,
	g *stagegraph.Graph,
	catalog Catalog,
	targetStageID string,
	rootInputs []domain.ArtifactInputIdentity,
	versions map[string]release.StageVersion,
	external map[string][]domain.ArtifactInputIdentity,
	models []domain.ArtifactInputIdentity,
) (Plan, error) {
	if len(rootInputs) == 0 {
		return Plan{}, fmt.Errorf("compile resolver requires immutable subject root inputs")
	}
	modelInputs, err := normalizeModelInputs(models)
	if err != nil {
		return Plan{}, err
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
		nodeSHA, err := stage.NodeSHA256()
		if err != nil {
			return Plan{}, err
		}
		if version.GraphNodeSHA256 != "" && version.GraphNodeSHA256 != nodeSHA {
			return Plan{}, fmt.Errorf("released stage graph identity drift: %s", stageID)
		}

		inputs := make([]domain.ArtifactInputIdentity, 0, len(rootInputs)+len(stage.DependsOn)+len(external[stageID])+len(modelInputs))
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
		for _, item := range external[stageID] {
			if item.Role == "" || item.ArtifactType == "" || item.SemanticSHA256 == "" {
				return Plan{}, fmt.Errorf("external stage input identity incomplete: %s", stageID)
			}
			inputs = append(inputs, domain.ArtifactInputIdentity{
				Role:           "external:" + item.Role,
				Ordinal:        len(inputs),
				ArtifactType:   item.ArtifactType,
				SemanticSHA256: item.SemanticSHA256,
			})
		}
		for _, item := range modelInputs {
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
				"graph_node_sha256": nodeSHA,
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
