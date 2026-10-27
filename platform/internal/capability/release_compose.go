package capability

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

// ProductProviders is the single source-level catalog of first-class product
// capabilities which are not already represented by Compiler stage nodes.
// New replaceable product boundaries (Mesh, Rig, Skin, ...) join here instead
// of inventing a second registry or a parallel operational graph. Catalog
// presence does not imply that a learned ModelRelease channel is admitted;
// execution remains target-scoped and fail-closed on live Platform selection.
func ProductProviders() []Provider {
	return []Provider{
		ProductInputProvider{},
		ProductShapeProvider{},
		ProductMeshProvider{},
		ProductRigProvider{},
		ProductSkinProvider{},
	}
}

func productProviderCatalog(ctx context.Context) ([]Descriptor, error) {
	var rows []Descriptor
	seen := map[string]struct{}{}
	for _, provider := range ProductProviders() {
		if provider == nil {
			return nil, fmt.Errorf("nil product capability provider")
		}
		descriptors, err := provider.Capabilities(ctx)
		if err != nil {
			return nil, err
		}
		for _, descriptor := range descriptors {
			if err := descriptor.Validate(); err != nil {
				return nil, err
			}
			if !strings.HasPrefix(descriptor.ID, "product.") {
				return nil, fmt.Errorf("PRODUCT_PROVIDER_NON_PRODUCT_CAPABILITY:%s", descriptor.ID)
			}
			if _, duplicate := seen[descriptor.ID]; duplicate {
				return nil, fmt.Errorf("duplicate capability id %s", descriptor.ID)
			}
			seen[descriptor.ID] = struct{}{}
			rows = append(rows, descriptor)
		}
	}
	// Catalog integrity is source-level authority and must remain complete even
	// when one exact EngineRelease can execute only a prefix/subset of the roles.
	if err := validateProductDescriptorRows(rows, true); err != nil {
		return nil, err
	}
	return rows, nil
}

func admittedProductDescriptors(ctx context.Context, stageRows []Descriptor) ([]Descriptor, error) {
	catalog, err := productProviderCatalog(ctx)
	if err != nil {
		return nil, err
	}
	availableStages := map[string]struct{}{}
	for _, descriptor := range stageRows {
		availableStages[descriptor.ID] = struct{}{}
	}
	byID := map[string]Descriptor{}
	for _, descriptor := range catalog {
		byID[descriptor.ID] = descriptor
	}

	memo := map[string]bool{}
	visiting := map[string]bool{}
	var unavailable func(string) (bool, error)
	unavailable = func(id string) (bool, error) {
		if value, ok := memo[id]; ok {
			return value, nil
		}
		if visiting[id] {
			return false, fmt.Errorf("PRODUCT_CAPABILITY_DEPENDENCY_CYCLE:%s", id)
		}
		descriptor, ok := byID[id]
		if !ok {
			return false, fmt.Errorf("PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:%s", id)
		}
		visiting[id] = true
		defer delete(visiting, id)
		for _, dependency := range descriptor.Dependencies {
			if _, ok := availableStages[dependency]; ok {
				continue
			}
			if strings.HasPrefix(dependency, "compiler.stage/") {
				memo[id] = true
				return true, nil
			}
			if strings.HasPrefix(dependency, "product.") {
				if _, declared := byID[dependency]; !declared {
					return false, fmt.Errorf("PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:%s:%s", id, dependency)
				}
				blocked, err := unavailable(dependency)
				if err != nil {
					return false, err
				}
				if blocked {
					memo[id] = true
					return true, nil
				}
				continue
			}
			return false, fmt.Errorf("PRODUCT_CAPABILITY_DEPENDENCY_UNKNOWN:%s:%s", id, dependency)
		}
		memo[id] = false
		return false, nil
	}

	admitted := make([]Descriptor, 0, len(catalog))
	for _, descriptor := range catalog {
		blocked, err := unavailable(descriptor.ID)
		if err != nil {
			return nil, err
		}
		if !blocked {
			admitted = append(admitted, descriptor)
		}
	}
	return admitted, nil
}

// BuildEngineReleaseRegistry composes the immutable release capability graph
// from the exact released Compiler DAG plus only the source-owned product roles
// whose declared dependencies are executable in that exact DAG. The source
// catalog remains complete; admission is a release projection, not role deletion.
func BuildEngineReleaseRegistry(ctx context.Context, graph *stagegraph.Graph) (*Registry, error) {
	if graph == nil {
		return nil, fmt.Errorf("release capability registry requires stage graph")
	}
	owners, err := architecture.Build(graph)
	if err != nil {
		return nil, err
	}
	stageRows, err := (CompilePlanProvider{Graph: graph, Architecture: owners}).Capabilities(ctx)
	if err != nil {
		return nil, err
	}
	productRows, err := admittedProductDescriptors(ctx, stageRows)
	if err != nil {
		return nil, err
	}
	rows := append(append([]Descriptor(nil), stageRows...), productRows...)
	registry, err := NewRegistry(rows)
	if err != nil {
		return nil, err
	}
	// product.* capability IDs are stable semantic roles. Release sealing is also
	// the architecture gate for the admitted role subset: shared semantics,
	// responsibility ownership, learned-model requirements, authority duplication,
	// transform paths and lossy boundaries remain fail-closed.
	if _, err := ProductRoleGraph(registry); err != nil {
		return nil, err
	}
	if err := validateReleaseProductDescriptorSemantics(registry); err != nil {
		return nil, err
	}
	if _, err := ResponsibilityTerrain(registry); err != nil {
		return nil, err
	}
	if _, err := ProductModelRequirements(registry); err != nil {
		return nil, err
	}
	return registry, nil
}

func modulePolicySHA256(descriptor Descriptor) (string, error) {
	return semantic.JSONSHA256(descriptorPolicyDocument(descriptor))
}

// SealEngineReleaseSnapshot seals exactly one capability-set identity for an
// existing immutable EngineRelease. Stage capability versions are derived from
// the already-sealed EngineRelease stage bindings. For admitted non-stage
// product capabilities, callers provide only implementation closure identities;
// Platform mints policy and parameter hashes from its exact admitted descriptor.
//
// This keeps EngineRelease DAG truth, learned ModelRelease requirements and
// capability-set truth separate but jointly immutable. A release cannot claim a
// product role whose Compiler prerequisites are absent, nor can a caller mint a
// second semantic identity for Platform-owned capability policy or parameters.
func SealEngineReleaseSnapshot(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	moduleImplementations map[string]ModuleImplementationIdentity,
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
	purpose, modelRequirements, err := release.LoadModelRequirements(ctx, pool, releaseID)
	if err != nil {
		return "", err
	}
	if err := validateEngineReleaseModelRequirements(registry, purpose, modelRequirements); err != nil {
		return "", fmt.Errorf("CAPABILITY_MODEL_REQUIREMENT_INVALID:%w", err)
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
			implementation, ok := moduleImplementations[descriptor.ID]
			if !ok {
				return "", fmt.Errorf("CAPABILITY_MODULE_IMPLEMENTATION_MISSING:%s", descriptor.ID)
			}
			version, err = mintModuleVersionIdentity(descriptor, implementation)
			if err != nil {
				return "", fmt.Errorf("CAPABILITY_MODULE_IMPLEMENTATION_INVALID:%s:%w", descriptor.ID, err)
			}
		}
		bindings = append(bindings, ReleaseBinding{Descriptor: descriptor, Version: version})
	}
	for capabilityID := range moduleImplementations {
		if _, ok := requiredModules[capabilityID]; !ok {
			return "", fmt.Errorf("CAPABILITY_MODULE_IMPLEMENTATION_UNKNOWN:%s", capabilityID)
		}
	}
	if len(moduleImplementations) != len(requiredModules) {
		return "", fmt.Errorf(
			"CAPABILITY_MODULE_IMPLEMENTATION_CARDINALITY_DRIFT:%d!=%d",
			len(moduleImplementations),
			len(requiredModules),
		)
	}
	return SealReleaseSnapshot(ctx, pool, releaseID, bindings, createdBy)
}
