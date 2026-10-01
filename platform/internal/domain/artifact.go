package domain

import (
	"errors"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type ArtifactInputIdentity struct {
	Role           string `json:"role"`
	Ordinal        int    `json:"ordinal"`
	ArtifactType   string `json:"artifact_type"`
	SemanticSHA256 string `json:"semantic_sha256"`
}

type ArtifactSemanticDescriptor struct {
	ArtifactType         string                  `json:"artifact_type"`
	SchemaVersion        string                  `json:"schema_version"`
	ProducerContract     string                  `json:"producer_contract"`
	ImplementationSHA256 string                  `json:"implementation_sha256"`
	PolicySHA256         string                  `json:"policy_sha256"`
	Inputs               []ArtifactInputIdentity `json:"inputs"`
	SemanticParameters   map[string]any          `json:"semantic_parameters"`
}

func (d ArtifactSemanticDescriptor) Validate() error {
	if d.ArtifactType == "" || d.SchemaVersion == "" || d.ProducerContract == "" {
		return errors.New("artifact type/schema/producer contract are required")
	}
	if err := semantic.ValidateSHA256(d.ImplementationSHA256); err != nil {
		return err
	}
	if err := semantic.ValidateSHA256(d.PolicySHA256); err != nil {
		return err
	}
	for i, in := range d.Inputs {
		if in.Role == "" || in.ArtifactType == "" || in.Ordinal != i {
			return errors.New("artifact inputs must be non-empty and contiguous in ordinal order")
		}
		if err := semantic.ValidateSHA256(in.SemanticSHA256); err != nil {
			return err
		}
	}
	if d.SemanticParameters == nil {
		d.SemanticParameters = map[string]any{}
	}
	return nil
}

func (d ArtifactSemanticDescriptor) SemanticSHA256() (string, error) {
	if err := d.Validate(); err != nil {
		return "", err
	}
	return semantic.JSONSHA256(d)
}

type ProductRoleBinding struct {
	Role                   string `json:"role"`
	ArtifactType           string `json:"artifact_type"`
	ArtifactSemanticSHA256 string `json:"artifact_semantic_sha256"`
}

type ProductRevisionManifest struct {
	ContractVersion string               `json:"contract_version"`
	SubjectID       string               `json:"subject_id"`
	Roles           []ProductRoleBinding `json:"roles"`
}

func (m ProductRevisionManifest) ManifestSHA256() (string, error) {
	if m.ContractVersion == "" {
		m.ContractVersion = "RealSaS.ProductRevisionManifest.v1"
	}
	if m.SubjectID == "" {
		return "", errors.New("subject_id is required")
	}
	last := ""
	seen := map[string]struct{}{}
	for _, role := range m.Roles {
		if role.Role == "" || role.ArtifactType == "" {
			return "", errors.New("product role and artifact type are required")
		}
		if _, ok := seen[role.Role]; ok || (last != "" && role.Role < last) {
			return "", errors.New("product roles must be unique and lexicographically sorted")
		}
		if err := semantic.ValidateSHA256(role.ArtifactSemanticSHA256); err != nil {
			return "", err
		}
		seen[role.Role] = struct{}{}
		last = role.Role
	}
	return semantic.JSONSHA256(m)
}
