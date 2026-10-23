package capability

import (
	"fmt"
	"sort"
	"strings"
)

// ProductRoleID is a stable semantic product boundary. The role survives
// implementation changes: a responsibility may move between learned models,
// Compiler code, runtime code, or a hybrid without renaming the product role.
type ProductRoleID string

const (
	ProductRoleInput     ProductRoleID = "input"
	ProductRoleShape     ProductRoleID = "shape"
	ProductRoleMesh      ProductRoleID = "mesh"
	ProductRoleRig       ProductRoleID = "rig"
	ProductRoleSkin      ProductRoleID = "skin"
	ProductRoleMechanics ProductRoleID = "mechanics"
	ProductRoleGraphics  ProductRoleID = "graphics"
	ProductRoleMotion    ProductRoleID = "motion"
	ProductRoleProof     ProductRoleID = "proof"
)

type ImplementationClass string

const (
	ImplementationModel    ImplementationClass = "MODEL"
	ImplementationCompiler ImplementationClass = "COMPILER"
	ImplementationHybrid   ImplementationClass = "HYBRID"
	ImplementationRuntime  ImplementationClass = "RUNTIME"
	ImplementationPlatform ImplementationClass = "PLATFORM"
)

// ProductRoleNode is the platform-facing view of one stable product role and
// the exact implementation selected by one capability registry/release.
type ProductRoleNode struct {
	RoleID                  ProductRoleID       `json:"role_id"`
	CapabilityID            string              `json:"capability_id"`
	ImplementationClass     ImplementationClass `json:"implementation_class"`
	OwnerModuleID           string              `json:"owner_module_id"`
	ExecutorActivity        string              `json:"executor_activity"`
	InterfaceContract       string              `json:"interface_contract"`
	ProductRoleDependencies []ProductRoleID     `json:"product_role_dependencies"`
	InternalDependencies    []string            `json:"internal_dependencies,omitempty"`
	ProductPassAuthority    bool                `json:"product_pass_authority"`
}

func validProductRoleID(value ProductRoleID) bool {
	switch value {
	case ProductRoleInput, ProductRoleShape, ProductRoleMesh, ProductRoleRig,
		ProductRoleSkin, ProductRoleMechanics, ProductRoleGraphics,
		ProductRoleMotion, ProductRoleProof:
		return true
	default:
		return false
	}
}

func validImplementationClass(value ImplementationClass) bool {
	switch value {
	case ImplementationModel, ImplementationCompiler, ImplementationHybrid,
		ImplementationRuntime, ImplementationPlatform:
		return true
	default:
		return false
	}
}

func productRoleMetadata(descriptor Descriptor) (ProductRoleID, ImplementationClass, string, bool, error) {
	if !strings.HasPrefix(descriptor.ID, "product.") {
		return "", "", "", false, nil
	}
	metadata := descriptor.Metadata
	role := ProductRoleID(fmt.Sprint(metadata["product_role_id"]))
	implementation := ImplementationClass(fmt.Sprint(metadata["implementation_class"]))
	contract := strings.TrimSpace(fmt.Sprint(metadata["interface_contract"]))
	if !validProductRoleID(role) {
		return "", "", "", true, fmt.Errorf("PRODUCT_ROLE_ID_INVALID:%s:%s", descriptor.ID, role)
	}
	if descriptor.ID != "product."+string(role) {
		return "", "", "", true, fmt.Errorf("PRODUCT_ROLE_CAPABILITY_ID_DRIFT:%s:%s", descriptor.ID, role)
	}
	if !validImplementationClass(implementation) {
		return "", "", "", true, fmt.Errorf("PRODUCT_ROLE_IMPLEMENTATION_CLASS_INVALID:%s:%s", descriptor.ID, implementation)
	}
	if contract == "" || contract == "<nil>" {
		return "", "", "", true, fmt.Errorf("PRODUCT_ROLE_INTERFACE_CONTRACT_MISSING:%s", descriptor.ID)
	}
	return role, implementation, contract, true, nil
}

// ProductRoleGraph projects an exact capability registry into the stable
// product-role graph. Dependencies on legacy Compiler stages or other internal
// capabilities remain visible as internal_dependencies rather than being
// mistaken for stable product roles.
func ProductRoleGraph(registry *Registry) ([]ProductRoleNode, error) {
	if registry == nil {
		return nil, fmt.Errorf("product role graph requires capability registry")
	}
	roleByCapability := map[string]ProductRoleID{}
	for _, descriptor := range registry.Descriptors() {
		role, _, _, isProductRole, err := productRoleMetadata(descriptor)
		if err != nil {
			return nil, err
		}
		if !isProductRole {
			continue
		}
		if _, exists := roleByCapability[descriptor.ID]; exists {
			return nil, fmt.Errorf("PRODUCT_ROLE_DUPLICATE_CAPABILITY:%s", descriptor.ID)
		}
		roleByCapability[descriptor.ID] = role
	}

	nodes := make([]ProductRoleNode, 0, len(roleByCapability))
	seenRoles := map[ProductRoleID]string{}
	for _, descriptor := range registry.Descriptors() {
		role, implementation, contract, isProductRole, err := productRoleMetadata(descriptor)
		if err != nil {
			return nil, err
		}
		if !isProductRole {
			continue
		}
		if prior, exists := seenRoles[role]; exists {
			return nil, fmt.Errorf("PRODUCT_ROLE_DUPLICATE_PROVIDER:%s:%s:%s", role, prior, descriptor.ID)
		}
		seenRoles[role] = descriptor.ID
		var productDeps []ProductRoleID
		var internalDeps []string
		for _, dependency := range descriptor.Dependencies {
			if depRole, ok := roleByCapability[dependency]; ok {
				productDeps = append(productDeps, depRole)
			} else {
				internalDeps = append(internalDeps, dependency)
			}
		}
		sort.Slice(productDeps, func(i, j int) bool { return productDeps[i] < productDeps[j] })
		sort.Strings(internalDeps)
		nodes = append(nodes, ProductRoleNode{
			RoleID:                  role,
			CapabilityID:            descriptor.ID,
			ImplementationClass:     implementation,
			OwnerModuleID:           descriptor.OwnerModuleID,
			ExecutorActivity:        descriptor.ExecutorActivity,
			InterfaceContract:       contract,
			ProductRoleDependencies: productDeps,
			InternalDependencies:    internalDeps,
			ProductPassAuthority:    descriptor.ProductPassAuthority,
		})
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].RoleID < nodes[j].RoleID })
	return nodes, nil
}
