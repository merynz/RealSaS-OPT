package capability

import (
	"fmt"
	"strings"
)

const (
	ProductRoleProjectionDeclared         = "DECLARED"
	ProductRoleProjectionLegacyUndeclared = "LEGACY_UNDECLARED"
)

// ProductRoleReadProjection keeps new release sealing strict while allowing
// historical snapshots, created before product-role metadata existed, to stay
// inspectable. A mixed declared/undeclared snapshot is rejected because it is
// ambiguous whether metadata was partially lost or a release was incompletely
// migrated.
func ProductRoleReadProjection(registry *Registry) ([]ProductRoleNode, string, error) {
	if registry == nil {
		return nil, "", fmt.Errorf("product role read projection requires capability registry")
	}
	productCount := 0
	declaredCount := 0
	for _, descriptor := range registry.Descriptors() {
		if !strings.HasPrefix(descriptor.ID, "product.") {
			continue
		}
		productCount++
		if descriptor.Metadata != nil {
			if raw, ok := descriptor.Metadata["product_role_id"]; ok && strings.TrimSpace(fmt.Sprint(raw)) != "" {
				declaredCount++
			}
		}
	}
	if productCount == 0 {
		return []ProductRoleNode{}, ProductRoleProjectionDeclared, nil
	}
	if declaredCount == 0 {
		return []ProductRoleNode{}, ProductRoleProjectionLegacyUndeclared, nil
	}
	if declaredCount != productCount {
		return nil, "", fmt.Errorf(
			"PRODUCT_ROLE_PROJECTION_MIXED_AUTHORITY_EPOCH:%d/%d",
			declaredCount,
			productCount,
		)
	}
	nodes, err := ProductRoleGraph(registry)
	if err != nil {
		return nil, "", err
	}
	return nodes, ProductRoleProjectionDeclared, nil
}
