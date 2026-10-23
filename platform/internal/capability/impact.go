package capability

import (
	"fmt"
	"sort"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type DirectCapabilityChange struct {
	CapabilityID          string        `json:"capability_id"`
	ProductRoleID         ProductRoleID `json:"product_role_id,omitempty"`
	ImplementationChanged bool          `json:"implementation_changed"`
	PolicyChanged         bool          `json:"policy_changed"`
	ParametersChanged     bool          `json:"parameters_changed"`
	DescriptorChanged     bool          `json:"descriptor_changed"`
	Added                 bool          `json:"added"`
	Removed               bool          `json:"removed"`
}

type CapabilityImpact struct {
	DirectChanges             []DirectCapabilityChange `json:"direct_changes"`
	InvalidatedCapabilityIDs  []string                 `json:"invalidated_capability_ids"`
	UnchangedCapabilityIDs    []string                 `json:"unchanged_capability_ids"`
	RemovedCapabilityIDs      []string                 `json:"removed_capability_ids"`
	ChangedProductRoleIDs     []ProductRoleID          `json:"changed_product_role_ids"`
	InvalidatedProductRoleIDs []ProductRoleID          `json:"invalidated_product_role_ids"`
}

func descriptorSHA256(value Descriptor) (string, error) {
	return semantic.JSONSHA256(value)
}

func descendantsIncluding(registry *Registry, roots ...string) ([]string, error) {
	if registry == nil {
		return nil, fmt.Errorf("capability registry is required")
	}
	children := map[string][]string{}
	for _, descriptor := range registry.Descriptors() {
		for _, dependency := range descriptor.Dependencies {
			children[dependency] = append(children[dependency], descriptor.ID)
		}
	}
	seen := map[string]struct{}{}
	queue := append([]string(nil), roots...)
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if _, ok := seen[id]; ok {
			continue
		}
		if _, ok := registry.Get(id); !ok {
			return nil, fmt.Errorf("unknown capability impact root %s", id)
		}
		seen[id] = struct{}{}
		for _, child := range children[id] {
			if _, ok := seen[child]; !ok {
				queue = append(queue, child)
			}
		}
	}
	ordered := make([]string, 0, len(seen))
	for _, descriptor := range registry.Descriptors() {
		if _, ok := seen[descriptor.ID]; ok {
			ordered = append(ordered, descriptor.ID)
		}
	}
	return ordered, nil
}

func productRoleForDescriptor(descriptor Descriptor) ProductRoleID {
	role, _, _, ok, err := productRoleMetadata(descriptor)
	if err != nil || !ok {
		return ""
	}
	return role
}

// CompareReleaseSnapshots compares exact immutable capability graphs. Stable
// product roles survive implementation movement; rewiring changes descriptor
// identity/version and invalidates downstream capabilities without forcing the
// semantic product role itself to be renamed.
func CompareReleaseSnapshots(oldSnapshot, newSnapshot ReleaseSnapshot) (CapabilityImpact, error) {
	if oldSnapshot.Registry == nil || newSnapshot.Registry == nil {
		return CapabilityImpact{}, fmt.Errorf("capability snapshots require registries")
	}
	oldDescriptors := map[string]Descriptor{}
	newDescriptors := map[string]Descriptor{}
	for _, descriptor := range oldSnapshot.Registry.Descriptors() {
		oldDescriptors[descriptor.ID] = descriptor
	}
	for _, descriptor := range newSnapshot.Registry.Descriptors() {
		newDescriptors[descriptor.ID] = descriptor
	}

	var direct []DirectCapabilityChange
	var removed []string
	var roots []string
	changedRoles := map[ProductRoleID]struct{}{}

	for id, oldDescriptor := range oldDescriptors {
		if _, exists := oldSnapshot.Versions[id]; !exists {
			return CapabilityImpact{}, fmt.Errorf("missing baseline capability version %s", id)
		}
		if _, survives := newDescriptors[id]; survives {
			continue
		}
		removed = append(removed, id)
		role := productRoleForDescriptor(oldDescriptor)
		direct = append(direct, DirectCapabilityChange{CapabilityID: id, ProductRoleID: role, Removed: true})
		if role != "" {
			changedRoles[role] = struct{}{}
		}
	}

	for id, descriptor := range newDescriptors {
		newVersion, ok := newSnapshot.Versions[id]
		if !ok {
			return CapabilityImpact{}, fmt.Errorf("missing candidate capability version %s", id)
		}
		oldDescriptor, existed := oldDescriptors[id]
		oldVersion, hadVersion := oldSnapshot.Versions[id]
		if existed && !hadVersion {
			return CapabilityImpact{}, fmt.Errorf("missing baseline capability version %s", id)
		}
		change := DirectCapabilityChange{
			CapabilityID:  id,
			ProductRoleID: productRoleForDescriptor(descriptor),
			Added:         !existed,
		}
		if existed {
			oldSHA, err := descriptorSHA256(oldDescriptor)
			if err != nil {
				return CapabilityImpact{}, err
			}
			newSHA, err := descriptorSHA256(descriptor)
			if err != nil {
				return CapabilityImpact{}, err
			}
			change.DescriptorChanged = oldSHA != newSHA
			change.ImplementationChanged = oldVersion.ImplementationSHA256 != newVersion.ImplementationSHA256
			change.PolicyChanged = oldVersion.PolicySHA256 != newVersion.PolicySHA256
			change.ParametersChanged = oldVersion.ParametersSHA256 != newVersion.ParametersSHA256
		}
		if change.Added || change.DescriptorChanged || change.ImplementationChanged || change.PolicyChanged || change.ParametersChanged {
			direct = append(direct, change)
			roots = append(roots, id)
			if change.ProductRoleID != "" {
				changedRoles[change.ProductRoleID] = struct{}{}
			}
		}
	}

	// If a removed capability had downstream consumers in the old graph, every
	// surviving consumer is a root in the new graph because its dependency
	// closure necessarily changed even if its own descriptor bytes were reused.
	if len(removed) > 0 {
		oldAffected, err := descendantsIncluding(oldSnapshot.Registry, removed...)
		if err != nil {
			return CapabilityImpact{}, err
		}
		for _, id := range oldAffected {
			if _, survives := newDescriptors[id]; survives {
				roots = append(roots, id)
			}
		}
	}

	rootSet := map[string]struct{}{}
	for _, id := range roots {
		rootSet[id] = struct{}{}
	}
	roots = roots[:0]
	for _, descriptor := range newSnapshot.Registry.Descriptors() {
		if _, ok := rootSet[descriptor.ID]; ok {
			roots = append(roots, descriptor.ID)
		}
	}
	var invalidated []string
	var err error
	if len(roots) > 0 {
		invalidated, err = descendantsIncluding(newSnapshot.Registry, roots...)
		if err != nil {
			return CapabilityImpact{}, err
		}
	}
	invalidSet := map[string]struct{}{}
	invalidatedRoles := map[ProductRoleID]struct{}{}
	for _, id := range invalidated {
		invalidSet[id] = struct{}{}
		if descriptor, ok := newDescriptors[id]; ok {
			if role := productRoleForDescriptor(descriptor); role != "" {
				invalidatedRoles[role] = struct{}{}
			}
		}
	}
	var unchanged []string
	for _, descriptor := range newSnapshot.Registry.Descriptors() {
		if _, changed := invalidSet[descriptor.ID]; !changed {
			unchanged = append(unchanged, descriptor.ID)
		}
	}
	sort.Strings(removed)
	sort.Slice(direct, func(i, j int) bool { return direct[i].CapabilityID < direct[j].CapabilityID })
	changedRoleRows := make([]ProductRoleID, 0, len(changedRoles))
	for role := range changedRoles {
		changedRoleRows = append(changedRoleRows, role)
	}
	sort.Slice(changedRoleRows, func(i, j int) bool { return changedRoleRows[i] < changedRoleRows[j] })
	invalidatedRoleRows := make([]ProductRoleID, 0, len(invalidatedRoles))
	for role := range invalidatedRoles {
		invalidatedRoleRows = append(invalidatedRoleRows, role)
	}
	sort.Slice(invalidatedRoleRows, func(i, j int) bool { return invalidatedRoleRows[i] < invalidatedRoleRows[j] })

	return CapabilityImpact{
		DirectChanges:             direct,
		InvalidatedCapabilityIDs:  invalidated,
		UnchangedCapabilityIDs:    unchanged,
		RemovedCapabilityIDs:      removed,
		ChangedProductRoleIDs:     changedRoleRows,
		InvalidatedProductRoleIDs: invalidatedRoleRows,
	}, nil
}
