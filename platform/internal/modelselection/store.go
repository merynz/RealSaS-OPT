package modelselection

import (
	"context"
	"sort"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

type Selection struct {
	Family                  string    `json:"family"`
	ModelReleaseID          uuid.UUID `json:"model_release_id"`
	ArchitectureID          string    `json:"architecture_id"`
	ModelEngineReleaseID    uuid.UUID `json:"model_engine_release_id"`
	SourceCodeSHA           string    `json:"source_code_sha"`
	CheckpointArtifactID    uuid.UUID `json:"checkpoint_artifact_id"`
	CheckpointContentSHA256 string    `json:"checkpoint_content_sha256"`
	Channel                 string    `json:"channel"`
	ChannelLockVersion      int64     `json:"channel_lock_version"`
	ArtifactType            string    `json:"-"`
	ArtifactSchemaVersion   string    `json:"-"`
	ArtifactSemanticSHA256  string    `json:"-"`
	ArtifactStorageKey      string    `json:"-"`
	ArtifactSizeBytes       int64     `json:"-"`
}

// CopyAttemptToProductRevisionTx preserves the immutable model snapshot already
// selected when the compile Attempt was accepted. It never resolves a channel.
func CopyAttemptToProductRevisionTx(ctx context.Context, tx pgx.Tx, attemptID, productRevisionID uuid.UUID) error {
	_, err := tx.Exec(ctx, `
		INSERT INTO product_revision_model_releases(product_revision_id,family,model_release_id)
		SELECT $2,family,model_release_id
		FROM attempt_model_releases
		WHERE attempt_id=$1
		ORDER BY family
	`, attemptID, productRevisionID)
	return err
}

// LoadAttempt reads only the immutable model snapshot bound at compile
// acceptance. No execution path is allowed to resolve a floating model channel.
func LoadAttempt(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) ([]Selection, error) {
	rows, err := pool.Query(ctx, `
		SELECT amr.family,amr.model_release_id,amr.selected_channel,amr.selected_channel_lock_version,
		       mr.architecture_id,mr.engine_release_id,mr.source_code_sha,
		       mr.checkpoint_artifact_id,a.content_sha256,
		       t.name,t.schema_version,a.semantic_sha256,a.storage_key,a.size_bytes
		FROM attempt_model_releases amr
		JOIN model_releases mr ON mr.id=amr.model_release_id
		JOIN artifacts a ON a.id=mr.checkpoint_artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE amr.attempt_id=$1
		ORDER BY amr.family
	`, attemptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Selection
	for rows.Next() {
		var selected Selection
		if err := rows.Scan(
			&selected.Family, &selected.ModelReleaseID, &selected.Channel, &selected.ChannelLockVersion,
			&selected.ArchitectureID, &selected.ModelEngineReleaseID, &selected.SourceCodeSHA,
			&selected.CheckpointArtifactID, &selected.CheckpointContentSHA256,
			&selected.ArtifactType, &selected.ArtifactSchemaVersion, &selected.ArtifactSemanticSHA256,
			&selected.ArtifactStorageKey, &selected.ArtifactSizeBytes,
		); err != nil {
			return nil, err
		}
		out = append(out, selected)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func IdentityRows(selections []Selection) []map[string]any {
	ordered := append([]Selection(nil), selections...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Family < ordered[j].Family })
	out := make([]map[string]any, 0, len(ordered))
	for _, selected := range ordered {
		out = append(out, map[string]any{
			"family":                    selected.Family,
			"model_release_id":          selected.ModelReleaseID.String(),
			"architecture_id":           selected.ArchitectureID,
			"checkpoint_artifact_id":    selected.CheckpointArtifactID.String(),
			"checkpoint_content_sha256": selected.CheckpointContentSHA256,
			"model_engine_release_id":   selected.ModelEngineReleaseID.String(),
			"source_code_sha":           selected.SourceCodeSHA,
			"selected_channel":          selected.Channel,
			"channel_lock_version":      selected.ChannelLockVersion,
		})
	}
	return out
}

func EngineRefs(selections []Selection) []orchestration.ModelReleaseRef {
	out := make([]orchestration.ModelReleaseRef, 0, len(selections))
	for _, selected := range selections {
		out = append(out, orchestration.ModelReleaseRef{
			Family:               selected.Family,
			ModelReleaseID:       selected.ModelReleaseID.String(),
			ArchitectureID:       selected.ArchitectureID,
			ModelEngineReleaseID: selected.ModelEngineReleaseID.String(),
			SourceCodeSHA:        selected.SourceCodeSHA,
			Checkpoint: orchestration.ArtifactRef{
				ID:             selected.CheckpointArtifactID.String(),
				Role:           "model:" + selected.Family,
				ArtifactType:   selected.ArtifactType,
				SchemaVersion:  selected.ArtifactSchemaVersion,
				SemanticSHA256: selected.ArtifactSemanticSHA256,
				StorageKey:     selected.ArtifactStorageKey,
				ContentSHA256:  selected.CheckpointContentSHA256,
				SizeBytes:      selected.ArtifactSizeBytes,
			},
		})
	}
	return out
}
