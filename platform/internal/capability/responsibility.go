package capability

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

//go:embed contracts/product_responsibilities_v1.json
var productResponsibilityLedgerJSON []byte

type ResponsibilityDefinition struct {
	ID                    string        `json:"id"`
	RoleID                ProductRoleID `json:"role_id"`
	RequiredWhenRolePresent bool        `json:"required_when_role_present"`
	AuthorityKind         string        `json:"authority_kind"`
	Exclusive             bool          `json:"exclusive"`
	CanonicalTransformID  string        `json:"canonical_transform_id"`
	DataReduction         string        `json:"data_reduction"`
	SourceTruthRetained   bool          `json:"source_truth_retained"`
}

type responsibilityRules struct {
	ExclusiveResponsibilityMayHaveOneActiveOwner bool `json:"exclusive_responsibility_may_have_one_active_owner"`
	LossyTransformMayNotReplaceSourceTruth       bool `json:"lossy_transform_may_not_replace_source_truth"`
	SourceTruthMustRemainAddressable              bool `json:"source_truth_must_remain_addressable_after_evidence_lowering"`
	LegacyStagesAreInternalImplementation         bool `json:"legacy_stages_are_internal_implementation_not_product_role_authority"`
}

type ResponsibilityLedger struct {
	Schema           string                     `json:"schema"`
	Rules            responsibilityRules        `json:"rules"`
	Responsibilities []ResponsibilityDefinition `json:"responsibilities"`
}

type ResponsibilityTerrainNode struct {
	ResponsibilityID      string              `json:"responsibility_id"`
	RoleID                ProductRoleID       `json:"role_id"`
	CapabilityID          string              `json:"capability_id"`
	ImplementationClass   ImplementationClass `json:"implementation_class"`
	OwnerModuleID         string              `json:"owner_module_id"`
	AuthorityKind         string              `json:"authority_kind"`
	Exclusive             bool                `json:"exclusive"`
	CanonicalTransformID  string              `json:"canonical_transform_id"`
	DataReduction         string              `json:"data_reduction"`
	SourceTruthRetained   bool                `json:"source_truth_retained"`
}

func loadResponsibilityLedger() (ResponsibilityLedger, error) {
	var value ResponsibilityLedger
	if err := json.Unmarshal(productResponsibilityLedgerJSON, &value); err != nil {
		return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_DECODE:%w", err)
	}
	if value.Schema != "RealSaS.ProductResponsibilityLedger.v1" {
		return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_SCHEMA:%s", value.Schema)
	}
	if !value.Rules.ExclusiveResponsibilityMayHaveOneActiveOwner ||
		!value.Rules.LossyTransformMayNotReplaceSourceTruth ||
		!value.Rules.SourceTruthMustRemainAddressable ||
		!value.Rules.LegacyStagesAreInternalImplementation {
		return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_RULES_INCOMPLETE")
	}
	seenIDs := map[string]struct{}{}
	seenTransforms := map[string]string{}
	for _, row := range value.Responsibilities {
		if strings.TrimSpace(row.ID) == "" || !validProductRoleID(row.RoleID) {
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_ENTRY_INVALID:%s", row.ID)
		}
		if _, exists := seenIDs[row.ID]; exists {
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_DUPLICATE_ID:%s", row.ID)
		}
		seenIDs[row.ID] = struct{}{}
		if strings.TrimSpace(row.CanonicalTransformID) == "" {
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_TRANSFORM_MISSING:%s", row.ID)
		}
		if prior, exists := seenTransforms[row.CanonicalTransformID]; exists && prior != row.ID {
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_TRANSFORM_REUSED:%s:%s:%s", row.CanonicalTransformID, prior, row.ID)
		}
		seenTransforms[row.CanonicalTransformID] = row.ID
		switch row.DataReduction {
		case "NONE", "LOSSLESS", "BOUNDED_LOSSY", "GENERATIVE_PROPOSAL":
		default:
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_DATA_REDUCTION_INVALID:%s:%s", row.ID, row.DataReduction)
		}
		if (row.DataReduction == "BOUNDED_LOSSY" || row.DataReduction == "GENERATIVE_PROPOSAL") && !row.SourceTruthRetained {
			return ResponsibilityLedger{}, fmt.Errorf("RESPONSIBILITY_LEDGER_LOSSY_REPLACES_SOURCE_TRUTH:%s", row.ID)
		}
	}
	return value, nil
}

func metadataStringList(metadata map[string]any, key string) ([]string, error) {
	raw, ok := metadata[key]
	if !ok {
		return nil, fmt.Errorf("RESPONSIBILITY_METADATA_MISSING:%s", key)
	}
	var rows []string
	switch value := raw.(type) {
	case []string:
		rows = append(rows, value...)
	case []any:
		for _, item := range value {
			text := strings.TrimSpace(fmt.Sprint(item))
			if text == "" || text == "<nil>" {
				return nil, fmt.Errorf("RESPONSIBILITY_METADATA_EMPTY:%s", key)
			}
			rows = append(rows, text)
		}
	default:
		return nil, fmt.Errorf("RESPONSIBILITY_METADATA_TYPE:%s:%T", key, raw)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("RESPONSIBILITY_METADATA_EMPTY:%s", key)
	}
	sort.Strings(rows)
	for i := 1; i < len(rows); i++ {
		if rows[i] == rows[i-1] {
			return nil, fmt.Errorf("RESPONSIBILITY_METADATA_DUPLICATE:%s:%s", key, rows[i])
		}
	}
	return rows, nil
}

// ResponsibilityTerrain projects the active product graph into the operational
// terrain map: stable responsibility, current implementation owner, authority
// kind, canonical transform path and data-reduction risk. Release sealing calls
// this fail-closed so accidental duplicate authority or lossy truth replacement
// cannot become a canonical product graph.
func ResponsibilityTerrain(registry *Registry) ([]ResponsibilityTerrainNode, error) {
	if registry == nil {
		return nil, fmt.Errorf("responsibility terrain requires capability registry")
	}
	ledger, err := loadResponsibilityLedger()
	if err != nil {
		return nil, err
	}
	definitions := map[string]ResponsibilityDefinition{}
	requiredByRole := map[ProductRoleID]map[string]struct{}{}
	for _, row := range ledger.Responsibilities {
		definitions[row.ID] = row
		if row.RequiredWhenRolePresent {
			if requiredByRole[row.RoleID] == nil {
				requiredByRole[row.RoleID] = map[string]struct{}{}
			}
			requiredByRole[row.RoleID][row.ID] = struct{}{}
		}
	}

	activeRoles := map[ProductRoleID]string{}
	owned := map[string]string{}
	ownedByRole := map[ProductRoleID]map[string]struct{}{}
	terrain := []ResponsibilityTerrainNode{}
	for _, descriptor := range registry.Descriptors() {
		role, implementation, _, isProductRole, err := productRoleMetadata(descriptor)
		if err != nil {
			return nil, err
		}
		if !isProductRole {
			continue
		}
		activeRoles[role] = descriptor.ID
		ids, err := metadataStringList(descriptor.Metadata, "responsibility_ids")
		if err != nil {
			return nil, fmt.Errorf("%s:%w", descriptor.ID, err)
		}
		if ownedByRole[role] == nil {
			ownedByRole[role] = map[string]struct{}{}
		}
		for _, id := range ids {
			definition, ok := definitions[id]
			if !ok {
				return nil, fmt.Errorf("RESPONSIBILITY_UNDECLARED:%s:%s", descriptor.ID, id)
			}
			if definition.RoleID != role {
				return nil, fmt.Errorf("RESPONSIBILITY_ROLE_DRIFT:%s:%s:%s", descriptor.ID, id, definition.RoleID)
			}
			if prior, exists := owned[id]; exists && definition.Exclusive {
				return nil, fmt.Errorf("RESPONSIBILITY_EXCLUSIVE_DUPLICATE_OWNER:%s:%s:%s", id, prior, descriptor.ID)
			}
			owned[id] = descriptor.ID
			ownedByRole[role][id] = struct{}{}
			if definition.AuthorityKind == "PRODUCT_PASS_AUTHORITY" && !descriptor.ProductPassAuthority {
				return nil, fmt.Errorf("RESPONSIBILITY_PRODUCT_PASS_OWNER_NOT_AUTHORIZED:%s", descriptor.ID)
			}
			terrain = append(terrain, ResponsibilityTerrainNode{
				ResponsibilityID: id,
				RoleID: role,
				CapabilityID: descriptor.ID,
				ImplementationClass: implementation,
				OwnerModuleID: descriptor.OwnerModuleID,
				AuthorityKind: definition.AuthorityKind,
				Exclusive: definition.Exclusive,
				CanonicalTransformID: definition.CanonicalTransformID,
				DataReduction: definition.DataReduction,
				SourceTruthRetained: definition.SourceTruthRetained,
			})
		}
	}
	for role := range activeRoles {
		for id := range requiredByRole[role] {
			if _, ok := ownedByRole[role][id]; !ok {
				return nil, fmt.Errorf("RESPONSIBILITY_REQUIRED_OWNER_MISSING:%s:%s", role, id)
			}
		}
	}
	sort.Slice(terrain, func(i, j int) bool {
		if terrain[i].RoleID != terrain[j].RoleID {
			return terrain[i].RoleID < terrain[j].RoleID
		}
		return terrain[i].ResponsibilityID < terrain[j].ResponsibilityID
	})
	return terrain, nil
}
