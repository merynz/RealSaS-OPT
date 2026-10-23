package capability

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

// ProductProviders is the single release-time list of first-class product
// capabilities which are not already represented by Compiler stage nodes.
// New replaceable product boundaries (Mesh, Rig, Skin, ...) join here instead
// of inventing a second registry or a parallel operational graph.
func ProductProviders() []Provider {
	return []Provider{
		ProductInputProvider{},
		ProductShapeProvider{},
	}
}

// BuildEngineReleaseRegistry composes the immutable release capability graph
// from the exact released Compiler DAG plus source-owned product capabilities.
func BuildEngineReleaseRegistry(ctx context.Context, graph *stagegraph.Graph) (*Registry, error) {
	if graph == nil {
		return nil, fmt.Errorf("release capability registry requires stage graph")
	}
	owners, err := architecture.Build(graph)
	if err != nil {
		return nil, err
	}
	providers := []Provider{CompilePlanProvider{Graph: graph, Architecture: owners}}
	providers = append(providers, ProductProviders()...)
	registry, err := Build(ctx, providers...)
	if err != nil {
		return nil, err
	}
	// product.* capability IDs are stable semantic roles. Fail release sealing
	// closed if role identity, implementation class, or role wiring drifts.
	if _, err := ProductRoleGraph(registry); err != nil {
		return nil, err
	}
	return registry, nil
}

func modulePolicySHA256(descriptor Descriptor) (string, error) {
	modes := make([]string, 0, len(descriptor.AllowedModes))
	for _, mode := range descriptor.AllowedModes {
		modes = append(modes, string(mode))
	}
	payload := map[string]any{
		"allowed_modes":          modes,
		"product_pass_authority": descriptor.ProductPassAuthority,
	}
	if len(descriptor.Dependencies) > 0 {
		payload["dependencies"] = append([]string(nil), descriptor.Dependencies...)
	}
	return semantic.JSONSHA256(payload)
}

func validateModuleVersionIdentity(descriptor Descriptor, version VersionIdentity) error {
	if err := version.Validate(); err != nil {
		return err
	}
	expectedPolicy, err := modulePolicySHA256(descriptor)
	if err != nil {
		return err
	}
	if version.PolicySHA256 != expectedPolicy {
		return fmt.Errorf("CAPABILITY_MODULE_POLICY_VERSION_DRIFT:%s", descriptor.ID)
	}
	expectedParameters, err := semantic.JSONSHA256(descriptor.Metadata)
	if err != nil {
		return err
	}
	if version.ParametersSHA256 != expectedParameters {
		return fmt.Errorf("CAPABILITY_MODULE_PARAMETERS_VERSION_DRIFT:%s", descriptor.ID)
	}
	return nil
}

// SealEngineReleaseSnapshot seals exactly one capability-set identity for an
// existing immutable EngineRelease. Stage capability versions are derived from
// the already-sealed EngineRelease stage bindings. Only non-stage product
// capabilities require an implementation identity from release tooling; policy
// and parameter hashes are independently rederived from the Go descriptor.
//
// This keeps EngineRelease DAG truth and capability-set truth separate but
// jointly immutable: either may be inspected independently, while developer
// capability execution fails closed if the capability snapshot is absent.
func SealEngineReleaseSnapshot(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	moduleVersions map[string]VersionIdentity,
	createdBy string,
) (string, error) {
	if pool == nil || releaseID == uuid.Nil || createdBy == "" {
		return "", fmt.Errorf("pool, release_id and created_by are required")
	}
	graph, _, err := release.LoadGraph(ctx, pool, releaseID)
	if err != nil {
		return "", err
	}
	registry, err := BuildEngineReleaseRegistry(ctx, graph)
	if err != nil {
		return "", err
	}
	stageVersions, err := release.LoadVersions(ctx, pool, releaseID, false)
	if err != nil {
		return "", err
	}

	bindings := make([]ReleaseBinding, 0, len(registry.Descriptors()))
	requiredModules := map[string]struct{}{}
	for _, descriptor := range registry.Descriptors() {
		var version VersionIdentity
		if descriptor.Kind == KindStage {
			stageID, _ := descriptor.Metadata["stage_id"].(string)
			stage, ok := stageVersions[stageID]
			if !ok || stageID == "" {
				return "", fmt.Errorf("CAPABILITY_STAGE_VERSION_MISSING:%s", descriptor.ID)
			}
			version = VersionIdentity{
				ImplementationSHA256: stage.ImplementationSHA256,
				PolicySHA256:         stage.PolicySHA256,
				ParametersSHA256:     stage.ParametersSHA256,
			}
		} else {
			requiredModules[descriptor.ID] = struct{}{}
			var ok bool
			version, ok = moduleVersions[descriptor.ID]
			if !ok {
				return "", fmt.Errorf("CAPABILITY_MODULE_VERSION_MISSING:%s", descriptor.ID)
			}
			if err := validateModuleVersionIdentity(descriptor, version); err != nil {
				return "", fmt.Errorf("CAPABILITY_MODULE_VERSION_INVALID:%s:%w", descriptor.ID, err)
			}
		}
		bindings = append(bindings, ReleaseBinding{Descriptor: descriptor, Version: version})
	}
	for capabilityID := range moduleVersions {
		if _, ok := requiredModules[capabilityID]; !ok {
			return "", fmt.Errorf("CAPABILITY_MODULE_VERSION_UNKNOWN:%s", capabilityID)
		}
	}
	if len(moduleVersions) != len(requiredModules) {
		return "", fmt.Errorf("CAPABILITY_MODULE_VERSION_CARDINALITY_DRIFT:%d!=%d", len(moduleVersions), len(requiredModules))
	}
	return SealReleaseSnapshot(ctx, pool, releaseID, bindings, createdBy)
}
