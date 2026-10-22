package modelselection

import (
	"context"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// LoadAttemptTx reads only the immutable model snapshot bound to an Attempt.
// It deliberately does not consult model_channels, so a later promotion cannot
// change a running compile or a ProductRevision being sealed.
func LoadAttemptTx(ctx context.Context, tx pgx.Tx, attemptID uuid.UUID) ([]Selection, error) {
	rows, err := tx.Query(ctx, `
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
