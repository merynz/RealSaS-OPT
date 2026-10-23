package capability

import (
	"fmt"
	"sort"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

type ProductModelRequirement struct {
	Family         string        `json:"family"`
	ArchitectureID string        `json:"architecture_id"`
	RoleID         ProductRoleID `json:"role_id"`
	CapabilityID   string        `json:"capability_id"`
}

func descriptorRequiredModels(descriptor Descriptor) ([]ProductModelRequirement, error) {
	role, _, _, isProductRole, err := productRoleMetadata(descriptor)
	if err != nil || !isProductRole {
		return nil, err
	}
	raw, ok := descriptor.Metadata["required_models"]
	if !ok {
		return []ProductModelRequirement{}, nil
	}
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENTS_TYPE:%s:%T", descriptor.ID, raw)
	}
	out := make([]ProductModelRequirement, 0, len(items))
	seen := map[string]struct{}{}
	for _, item := range items {
		row, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENT_ENTRY_TYPE:%s:%T", descriptor.ID, item)
		}
		family := strings.TrimSpace(fmt.Sprint(row["family"]))
		architectureID := strings.TrimSpace(fmt.Sprint(row["architecture_id"]))
		if family == "" || architectureID == "" || family == "<nil>" || architectureID == "<nil>" {
			return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENT_INVALID:%s", descriptor.ID)
		}
		if _, exists := seen[family]; exists {
			return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENT_DUPLICATE_FAMILY:%s:%s", descriptor.ID, family)
		}
		seen[family] = struct{}{}
		out = append(out, ProductModelRequirement{
			Family: family, ArchitectureID: architectureID, RoleID: role, CapabilityID: descriptor.ID,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })
	return out, nil
}

// ProductModelRequirements couples learned runtime dependencies to stable roles.
// Responsibility terrain decides which families are actually required; the
// capability semantic contract supplies each exact architecture identity.
func ProductModelRequirements(registry *Registry) ([]ProductModelRequirement, error) {
	if registry == nil {
		return nil, fmt.Errorf("product model requirements registry required")
	}
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		return nil, err
	}
	requiredByRole := map[ProductRoleID]map[string]struct{}{}
	for _, row := range terrain {
		if row.ModelFamily == "" {
			continue
		}
		if requiredByRole[row.RoleID] == nil {
			requiredByRole[row.RoleID] = map[string]struct{}{}
		}
		requiredByRole[row.RoleID][row.ModelFamily] = struct{}{}
	}

	globalArchitecture := map[string]string{}
	globalRow := map[string]ProductModelRequirement{}
	for _, descriptor := range registry.Descriptors() {
		role, _, _, isProductRole, err := productRoleMetadata(descriptor)
		if err != nil {
			return nil, err
		}
		if !isProductRole {
			continue
		}
		declared, err := descriptorRequiredModels(descriptor)
		if err != nil {
			return nil, err
		}
		declaredSet := map[string]struct{}{}
		for _, row := range declared {
			declaredSet[row.Family] = struct{}{}
			if prior, exists := globalArchitecture[row.Family]; exists && prior != row.ArchitectureID {
				return nil, fmt.Errorf("PRODUCT_MODEL_ARCHITECTURE_DRIFT:%s:%s!=%s", row.Family, prior, row.ArchitectureID)
			}
			globalArchitecture[row.Family] = row.ArchitectureID
			if _, exists := globalRow[row.Family]; !exists {
				globalRow[row.Family] = row
			}
		}
		for family := range requiredByRole[role] {
			if _, ok := declaredSet[family]; !ok {
				return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENT_MISSING:%s:%s", descriptor.ID, family)
			}
		}
		for family := range declaredSet {
			if _, ok := requiredByRole[role][family]; !ok {
				return nil, fmt.Errorf("PRODUCT_MODEL_REQUIREMENT_WITHOUT_RESPONSIBILITY:%s:%s", descriptor.ID, family)
			}
		}
	}

	out := make([]ProductModelRequirement, 0, len(globalRow))
	for _, row := range globalRow {
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })
	return out, nil
}

func validateEngineReleaseModelRequirements(registry *Registry, purpose release.Purpose, actual []release.ModelRequirement) error {
	expected, err := ProductModelRequirements(registry)
	if err != nil {
		return err
	}
	actualByFamily := map[string]release.ModelRequirement{}
	for _, row := range actual {
		if _, exists := actualByFamily[row.Family]; exists {
			return fmt.Errorf("ENGINE_RELEASE_MODEL_REQUIREMENT_DUPLICATE:%s", row.Family)
		}
		actualByFamily[row.Family] = row
	}
	if len(actualByFamily) != len(expected) {
		return fmt.Errorf("ENGINE_RELEASE_MODEL_REQUIREMENT_CARDINALITY:%d!=%d", len(actualByFamily), len(expected))
	}
	for _, want := range expected {
		got, ok := actualByFamily[want.Family]
		if !ok {
			return fmt.Errorf("ENGINE_RELEASE_MODEL_REQUIREMENT_MISSING:%s", want.Family)
		}
		if got.ArchitectureID != want.ArchitectureID {
			return fmt.Errorf("ENGINE_RELEASE_MODEL_ARCHITECTURE_DRIFT:%s:%s!=%s", want.Family, got.ArchitectureID, want.ArchitectureID)
		}
		if purpose == release.PurposeProduct && got.Channel != "PRODUCTION" {
			return fmt.Errorf("ENGINE_RELEASE_PRODUCT_MODEL_CHANNEL:%s:%s", want.Family, got.Channel)
		}
		if purpose == release.PurposeResearch && got.Channel != "RESEARCH" && got.Channel != "CANDIDATE" {
			return fmt.Errorf("ENGINE_RELEASE_RESEARCH_MODEL_CHANNEL:%s:%s", want.Family, got.Channel)
		}
	}
	return nil
}
