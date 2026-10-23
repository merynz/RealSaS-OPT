package capability

import (
	"fmt"
	"strings"
)

const (
	ResponsibilityProjectionDeclared         = "DECLARED"
	ResponsibilityProjectionLegacyUndeclared = "LEGACY_UNDECLARED"
)

// ResponsibilityReadProjection keeps new release sealing strict while allowing
// capability snapshots created before the responsibility terrain contract to
// remain inspectable. Mixed metadata is rejected because partial terrain data is
// more dangerous than clearly-declared historical absence.
func ResponsibilityReadProjection(registry *Registry) ([]ResponsibilityTerrainNode, string, error) {
	if registry == nil {
		return nil, "", fmt.Errorf("responsibility read projection requires capability registry")
	}
	productCount := 0
	declaredCount := 0
	for _, descriptor := range registry.Descriptors() {
		if !strings.HasPrefix(descriptor.ID, "product.") {
			continue
		}
		productCount++
		if descriptor.Metadata != nil {
			if _, ok := descriptor.Metadata["responsibility_ids"]; ok {
				declaredCount++
			}
		}
	}
	if productCount == 0 {
		return []ResponsibilityTerrainNode{}, ResponsibilityProjectionDeclared, nil
	}
	if declaredCount == 0 {
		return []ResponsibilityTerrainNode{}, ResponsibilityProjectionLegacyUndeclared, nil
	}
	if declaredCount != productCount {
		return nil, "", fmt.Errorf(
			"RESPONSIBILITY_PROJECTION_MIXED_AUTHORITY_EPOCH:%d/%d",
			declaredCount,
			productCount,
		)
	}
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		return nil, "", err
	}
	return terrain, ResponsibilityProjectionDeclared, nil
}
