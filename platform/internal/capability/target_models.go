package capability

import (
	"fmt"
	"sort"
)

// ResolvedProductModelRequirements projects one already-resolved capability
// closure into only the learned model identities that closure actually consumes.
//
// EngineRelease keeps the superset of model requirements for every admitted
// product role. A modular DEVELOPER_RUN must not snapshot unrelated model
// channels merely because another admitted role exists in the same release.
// Product compile candidates remain release-wide and are intentionally governed
// by the database product-attempt trigger.
func ResolvedProductModelRequirements(descriptors []Descriptor) ([]ProductModelRequirement, error) {
	byFamily := map[string]ProductModelRequirement{}
	for _, descriptor := range descriptors {
		_, _, _, isProductRole, err := productRoleMetadata(descriptor)
		if err != nil {
			return nil, err
		}
		if !isProductRole {
			continue
		}
		rows, err := descriptorRequiredModels(descriptor)
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			if prior, exists := byFamily[row.Family]; exists {
				if prior.ArchitectureID != row.ArchitectureID {
					return nil, fmt.Errorf(
						"RESOLVED_PRODUCT_MODEL_ARCHITECTURE_DRIFT:%s:%s!=%s",
						row.Family, prior.ArchitectureID, row.ArchitectureID,
					)
				}
				continue
			}
			byFamily[row.Family] = row
		}
	}
	out := make([]ProductModelRequirement, 0, len(byFamily))
	for _, row := range byFamily {
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })
	return out, nil
}
