package capability

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

//go:embed contracts/product_capabilities_v1.json
var productCapabilitySemanticsJSON []byte

type productCapabilitySemantic struct {
	EntryModule string         `json:"entry_module"`
	Policy      map[string]any `json:"policy"`
	Metadata    map[string]any `json:"metadata"`
}

type productCapabilitySemantics struct {
	Schema       string                               `json:"schema"`
	Capabilities map[string]productCapabilitySemantic `json:"capabilities"`
}

func loadProductCapabilitySemantics() (productCapabilitySemantics, error) {
	var value productCapabilitySemantics
	if err := json.Unmarshal(productCapabilitySemanticsJSON, &value); err != nil {
		return productCapabilitySemantics{}, fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_DECODE:%w", err)
	}
	if value.Schema != "RealSaS.ProductCapabilitySemantics.v1" {
		return productCapabilitySemantics{}, fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_SCHEMA:%s", value.Schema)
	}
	if len(value.Capabilities) == 0 {
		return productCapabilitySemantics{}, fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_EMPTY")
	}
	for id, row := range value.Capabilities {
		if !strings.HasPrefix(id, "product.") || row.EntryModule == "" || row.Policy == nil || row.Metadata == nil {
			return productCapabilitySemantics{}, fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_INVALID:%s", id)
		}
	}
	return value, nil
}

func descriptorPolicyDocument(descriptor Descriptor) map[string]any {
	modes := make([]string, 0, len(descriptor.AllowedModes))
	for _, mode := range descriptor.AllowedModes {
		modes = append(modes, string(mode))
	}
	value := map[string]any{
		"allowed_modes":          modes,
		"product_pass_authority": descriptor.ProductPassAuthority,
	}
	if len(descriptor.Dependencies) > 0 {
		value["dependencies"] = append([]string(nil), descriptor.Dependencies...)
	}
	return value
}

func descriptorSemanticMetadata(descriptor Descriptor) map[string]any {
	value := map[string]any{}
	for key, item := range descriptor.Metadata {
		if key == "responsibility_ids" {
			continue
		}
		value[key] = item
	}
	return value
}

func validateProductDescriptorRows(descriptors []Descriptor, requireFullCatalog bool) error {
	semantics, err := loadProductCapabilitySemantics()
	if err != nil {
		return err
	}
	seen := map[string]struct{}{}
	for _, descriptor := range descriptors {
		row, declared := semantics.Capabilities[descriptor.ID]
		if !declared {
			if strings.HasPrefix(descriptor.ID, "product.") {
				return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_UNDECLARED_PROVIDER:%s", descriptor.ID)
			}
			continue
		}
		if descriptor.Kind != KindModule {
			return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_KIND_DRIFT:%s", descriptor.ID)
		}
		if _, duplicate := seen[descriptor.ID]; duplicate {
			return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_DUPLICATE_PROVIDER:%s", descriptor.ID)
		}
		descriptorPolicySHA, err := semantic.JSONSHA256(descriptorPolicyDocument(descriptor))
		if err != nil {
			return err
		}
		contractPolicySHA, err := semantic.JSONSHA256(row.Policy)
		if err != nil {
			return err
		}
		if descriptorPolicySHA != contractPolicySHA {
			return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_POLICY_DRIFT:%s", descriptor.ID)
		}
		descriptorMetadataSHA, err := semantic.JSONSHA256(descriptorSemanticMetadata(descriptor))
		if err != nil {
			return err
		}
		contractMetadataSHA, err := semantic.JSONSHA256(row.Metadata)
		if err != nil {
			return err
		}
		if descriptorMetadataSHA != contractMetadataSHA {
			return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_METADATA_DRIFT:%s", descriptor.ID)
		}
		seen[descriptor.ID] = struct{}{}
	}
	if requireFullCatalog && len(seen) != len(semantics.Capabilities) {
		missing := make([]string, 0)
		for id := range semantics.Capabilities {
			if _, ok := seen[id]; !ok {
				missing = append(missing, id)
			}
		}
		sort.Strings(missing)
		return fmt.Errorf("PRODUCT_CAPABILITY_SEMANTICS_PROVIDER_MISSING:%v", missing)
	}
	return nil
}

// validateProductDescriptorSemantics protects the source-level product catalog:
// every declared product role must have exactly one provider and must match the
// shared semantic contract. Exact EngineReleases may later admit only a subset
// whose dependencies are executable in that release graph.
func validateProductDescriptorSemantics(registry *Registry) error {
	if registry == nil {
		return fmt.Errorf("product descriptor semantics registry required")
	}
	return validateProductDescriptorRows(registry.Descriptors(), true)
}

func validateReleaseProductDescriptorSemantics(registry *Registry) error {
	if registry == nil {
		return fmt.Errorf("release product descriptor semantics registry required")
	}
	return validateProductDescriptorRows(registry.Descriptors(), false)
}
