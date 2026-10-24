package capability

import (
	"fmt"

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
