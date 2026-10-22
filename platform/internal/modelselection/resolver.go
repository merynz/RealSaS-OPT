package modelselection

import (
	"context"
	"encoding/hex"
	"errors"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

const ResolverArtifactType = "RealSaS.ModelReleaseSelection"

type behaviorIdentity struct {
	Schema                  string `json:"schema"`
	Family                  string `json:"family"`
	ModelReleaseID          string `json:"model_release_id"`
	ArchitectureID          string `json:"architecture_id"`
	ModelEngineReleaseID    string `json:"model_engine_release_id"`
	SourceCodeSHA           string `json:"source_code_sha"`
	CheckpointArtifactID    string `json:"checkpoint_artifact_id"`
	CheckpointContentSHA256 string `json:"checkpoint_content_sha256"`
}

func validateSourceSHA(value string) error {
	if len(value) != 40 || strings.ToLower(value) != value {
		return errors.New("MODEL_SELECTION_SOURCE_SHA_INVALID")
	}
	raw, err := hex.DecodeString(value)
	if err != nil || len(raw) != 20 {
		return errors.New("MODEL_SELECTION_SOURCE_SHA_INVALID")
	}
	return nil
}

func validateFamily(value string) error {
	if value == "" || value != strings.TrimSpace(value) {
		return errors.New("MODEL_SELECTION_FAMILY_INVALID")
	}
	for _, ch := range value {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' {
			continue
		}
		return errors.New("MODEL_SELECTION_FAMILY_INVALID")
	}
	return nil
}

// BehaviorIdentitySHA256 is the exact behavior-relevant model identity used by
// the compile resolver/cache. Channel and lock-version provenance are
// intentionally excluded: moving a channel away and back to the same immutable
// ModelRelease must not manufacture a different inference behavior identity.
func BehaviorIdentitySHA256(selected Selection) (string, error) {
	if err := validateFamily(selected.Family); err != nil {
		return "", err
	}
	if selected.ModelReleaseID == uuid.Nil || selected.ModelEngineReleaseID == uuid.Nil || selected.CheckpointArtifactID == uuid.Nil {
		return "", errors.New("MODEL_SELECTION_IDENTITY_ID_REQUIRED")
	}
	if selected.ArchitectureID == "" || selected.ArchitectureID != strings.TrimSpace(selected.ArchitectureID) {
		return "", errors.New("MODEL_SELECTION_ARCHITECTURE_INVALID")
	}
	if err := validateSourceSHA(selected.SourceCodeSHA); err != nil {
		return "", err
	}
	if err := semantic.ValidateSHA256(selected.CheckpointContentSHA256); err != nil {
		return "", errors.New("MODEL_SELECTION_CHECKPOINT_SHA_INVALID")
	}
	return semantic.JSONSHA256(behaviorIdentity{
		Schema:                  "RealSaS.ModelReleaseBehaviorIdentity.v1",
		Family:                  selected.Family,
		ModelReleaseID:          selected.ModelReleaseID.String(),
		ArchitectureID:          selected.ArchitectureID,
		ModelEngineReleaseID:    selected.ModelEngineReleaseID.String(),
		SourceCodeSHA:           selected.SourceCodeSHA,
		CheckpointArtifactID:    selected.CheckpointArtifactID.String(),
		CheckpointContentSHA256: selected.CheckpointContentSHA256,
	})
}

func ResolverInputsFromSelections(selections []Selection) ([]domain.ArtifactInputIdentity, error) {
	ordered := append([]Selection(nil), selections...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Family < ordered[j].Family })
	out := make([]domain.ArtifactInputIdentity, 0, len(ordered))
	prior := ""
	for index, selected := range ordered {
		if index > 0 && selected.Family == prior {
			return nil, errors.New("MODEL_SELECTION_DUPLICATE_FAMILY")
		}
		digest, err := BehaviorIdentitySHA256(selected)
		if err != nil {
			return nil, err
		}
		out = append(out, domain.ArtifactInputIdentity{
			Role:           "model:" + selected.Family,
			Ordinal:        index,
			ArtifactType:   ResolverArtifactType,
			SemanticSHA256: digest,
		})
		prior = selected.Family
	}
	return out, nil
}

// ResolverInputs reads only the immutable Attempt snapshot selected at compile
// acceptance. It never resolves a live model channel.
func ResolverInputs(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) ([]domain.ArtifactInputIdentity, error) {
	selected, err := LoadAttempt(ctx, pool, attemptID)
	if err != nil {
		return nil, err
	}
	return ResolverInputsFromSelections(selected)
}
