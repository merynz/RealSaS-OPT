package domain

import (
	"errors"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type RenderRequestSpec struct {
	ContractVersion               string         `json:"contract_version"`
	ProductRevisionManifestSHA256 string         `json:"product_revision_manifest_sha256"`
	MotionArtifactSemanticSHA256  string         `json:"motion_artifact_semantic_sha256"`
	ViewSpec                      map[string]any `json:"view_spec"`
	RenderSettings                map[string]any `json:"render_settings"`
}

func (s RenderRequestSpec) normalized() RenderRequestSpec {
	if s.ContractVersion == "" {
		s.ContractVersion = "RealSaS.RenderRequest.v1"
	}
	if s.ViewSpec == nil {
		s.ViewSpec = map[string]any{}
	}
	if s.RenderSettings == nil {
		s.RenderSettings = map[string]any{}
	}
	return s
}

func (s RenderRequestSpec) SemanticSHA256() (string, error) {
	s = s.normalized()
	if err := semantic.ValidateSHA256(s.ProductRevisionManifestSHA256); err != nil {
		return "", errors.New("invalid product revision manifest sha256")
	}
	if err := semantic.ValidateSHA256(s.MotionArtifactSemanticSHA256); err != nil {
		return "", errors.New("invalid motion artifact semantic sha256")
	}
	return semantic.JSONSHA256(s)
}
