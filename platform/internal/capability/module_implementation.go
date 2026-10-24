package capability

import (
	"fmt"
	"sort"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

// ModuleImplementationIdentity is the only non-stage capability identity a
// release caller may assert. Policy and parameter identities belong to the
// exact admitted Platform descriptor and are minted server-side when the
// EngineRelease capability snapshot is sealed.
type ModuleImplementationIdentity struct {
	ImplementationSHA256 string `json:"implementation_sha256"`
}

func (v ModuleImplementationIdentity) Validate() error {
	if err := semantic.ValidateSHA256(v.ImplementationSHA256); err != nil {
		return fmt.Errorf("implementation sha: %w", err)
	}
	return nil
}

type moduleImplementationBinding struct {
	CapabilityID         string `json:"capability_id"`
	ImplementationSHA256 string `json:"implementation_sha256"`
}

// ModuleImplementationSetSHA256 mints the Platform-owned identity of the
// source-owned executable implementation set supplied for first-class product
// capabilities. Policy/parameter identities are intentionally excluded: those
// remain descriptor-owned and are sealed in the capability snapshot.
func ModuleImplementationSetSHA256(values map[string]ModuleImplementationIdentity) (string, error) {
	if len(values) == 0 {
		return "", fmt.Errorf("capability implementation set cannot be empty")
	}
	ids := make([]string, 0, len(values))
	for id, value := range values {
		if !strings.HasPrefix(id, "product.") {
			return "", fmt.Errorf("CAPABILITY_MODULE_IMPLEMENTATION_ID_INVALID:%s", id)
		}
		if err := value.Validate(); err != nil {
			return "", fmt.Errorf("%s: %w", id, err)
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	bindings := make([]moduleImplementationBinding, 0, len(ids))
	for _, id := range ids {
		bindings = append(bindings, moduleImplementationBinding{
			CapabilityID:         id,
			ImplementationSHA256: values[id].ImplementationSHA256,
		})
	}
	return semantic.JSONSHA256(bindings)
}

// BindReleaseManifestImplementationIdentity prevents a caller from asserting a
// second digest authority. The API computes the exact implementation-set hash
// from the supplied source-owned identities and injects it before release.Seal,
// making EngineRelease reuse sensitive to executable product-code changes.
func BindReleaseManifestImplementationIdentity(
	manifest *release.Manifest,
	values map[string]ModuleImplementationIdentity,
) error {
	if manifest == nil {
		return fmt.Errorf("release manifest is required")
	}
	if manifest.ContractVersion != "RealSaS.EngineReleaseManifest.v2" {
		if manifest.CapabilityImplementationsSHA256 != "" {
			return fmt.Errorf("CAPABILITY_IMPLEMENTATION_BINDING_V2_ONLY")
		}
		return nil
	}
	if manifest.CapabilityImplementationsSHA256 != "" {
		return fmt.Errorf("CAPABILITY_IMPLEMENTATION_BINDING_CALLER_ASSERTION_FORBIDDEN")
	}
	digest, err := ModuleImplementationSetSHA256(values)
	if err != nil {
		return err
	}
	manifest.CapabilityImplementationsSHA256 = digest
	return nil
}

func mintModuleVersionIdentity(
	descriptor Descriptor,
	implementation ModuleImplementationIdentity,
) (VersionIdentity, error) {
	if err := implementation.Validate(); err != nil {
		return VersionIdentity{}, err
	}
	policySHA, err := modulePolicySHA256(descriptor)
	if err != nil {
		return VersionIdentity{}, err
	}
	parametersSHA, err := semantic.JSONSHA256(descriptor.Metadata)
	if err != nil {
		return VersionIdentity{}, err
	}
	return VersionIdentity{
		ImplementationSHA256: implementation.ImplementationSHA256,
		PolicySHA256:         policySHA,
		ParametersSHA256:     parametersSHA,
	}, nil
}
