package capability

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

//go:embed contracts/product_capabilities_v1.json
var productCapabilityContractsJSON []byte

type productCapabilityContract struct {
	EntryModule string         `json:"entry_module"`
	Policy      map[string]any `json:"policy"`
	Metadata    map[string]any `json:"metadata"`
}

type productCapabilityContracts struct {
	Schema       string                               `json:"schema"`
	Capabilities map[string]productCapabilityContract `json:"capabilities"`
}

func loadProductCapabilityContracts() (productCapabilityContracts, error) {
	var value productCapabilityContracts
	if err := json.Unmarshal(productCapabilityContractsJSON, &value); err != nil {
		return productCapabilityContracts{}, fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_DECODE:%w", err)
	}
	if value.Schema != "RealSaS.ProductCapabilityContracts.v1" {
		return productCapabilityContracts{}, fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_SCHEMA:%s", value.Schema)
	}
	if len(value.Capabilities) == 0 {
		return productCapabilityContracts{}, fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_EMPTY")
	}
	for id, contract := range value.Capabilities {
		if id == "" || contract.EntryModule == "" || contract.Policy == nil || contract.Metadata == nil {
			return productCapabilityContracts{}, fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_INCOMPLETE:%s", id)
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

func validateProductDescriptorContracts(registry *Registry) error {
	if registry == nil {
		return fmt.Errorf("product descriptor contract registry required")
	}
	contracts, err := loadProductCapabilityContracts()
	if err != nil {
		return err
	}
	seen := map[string]struct{}{}
	for _, descriptor := range registry.Descriptors() {
		if _, ok := contracts.Capabilities[descriptor.ID]; !ok {
			continue
		}
		contract := contracts.Capabilities[descriptor.ID]
		if descriptor.Kind != KindModule {
			return fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_KIND_DRIFT:%s", descriptor.ID)
		}
		descriptorPolicySHA, err := semantic.JSONSHA256(descriptorPolicyDocument(descriptor))
		if err != nil {
			return err
		}
		contractPolicySHA, err := semantic.JSONSHA256(contract.Policy)
		if err != nil {
			return err
		}
		if descriptorPolicySHA != contractPolicySHA {
			return fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_POLICY_DRIFT:%s", descriptor.ID)
		}
		descriptorMetadataSHA, err := semantic.JSONSHA256(descriptor.Metadata)
		if err != nil {
			return err
		}
		contractMetadataSHA, err := semantic.JSONSHA256(contract.Metadata)
		if err != nil {
			return err
		}
		if descriptorMetadataSHA != contractMetadataSHA {
			return fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_METADATA_DRIFT:%s", descriptor.ID)
		}
		seen[descriptor.ID] = struct{}{}
	}
	if len(seen) != len(contracts.Capabilities) {
		missing := make([]string, 0)
		for id := range contracts.Capabilities {
			if _, ok := seen[id]; !ok {
				missing = append(missing, id)
			}
		}
		sort.Strings(missing)
		return fmt.Errorf("PRODUCT_CAPABILITY_CONTRACT_PROVIDER_MISSING:%v", missing)
	}
	return nil
}
