package modelselection

import (
	"context"
	"errors"
	"fmt"
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

func ResolveForEngineReleaseTx(ctx context.Context, tx pgx.Tx, engineReleaseID uuid.UUID) ([]Selection, error) {
	rows, err := tx.Query(ctx, `
		SELECT family,architecture_id,channel
		FROM engine_release_model_requirements
		WHERE release_id=$1
		ORDER BY family
	`, engineReleaseID)
	if err != nil {
		return nil, err
	}
	type requirement struct{ family, architecture, channel string }
	var requirements []requirement
	for rows.Next() {
		var r requirement
		if err := rows.Scan(&r.family, &r.architecture, &r.channel); err != nil {
			rows.Close()
			return nil, err
		}
		requirements = append(requirements, r)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, err
	}
	rows.Close()

	selections := make([]Selection, 0, len(requirements))
	for _, requirement := range requirements {
		var selected Selection
		var lifecycle, artifactContentSHA string
		selected.Family = requirement.family
		selected.Channel = requirement.channel
		err := tx.QueryRow(ctx, `
			SELECT mc.model_release_id,mc.lock_version,
			       mr.architecture_id,mr.engine_release_id,mr.source_code_sha,mr.lifecycle_state,
			       mr.checkpoint_artifact_id,mr.checkpoint_content_sha256,
			       t.name,t.schema_version,a.semantic_sha256,a.storage_key,a.content_sha256,a.size_bytes
			FROM model_channels mc
			JOIN model_releases mr ON mr.id=mc.model_release_id
			JOIN artifacts a ON a.id=mr.checkpoint_artifact_id
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE mc.family=$1 AND mc.channel=$2
		`, requirement.family, requirement.channel).Scan(
			&selected.ModelReleaseID, &selected.ChannelLockVersion,
			&selected.ArchitectureID, &selected.ModelEngineReleaseID, &selected.SourceCodeSHA, &lifecycle,
			&selected.CheckpointArtifactID, &selected.CheckpointContentSHA256,
			&selected.ArtifactType, &selected.ArtifactSchemaVersion, &selected.ArtifactSemanticSHA256,
			&selected.ArtifactStorageKey, &artifactContentSHA, &selected.ArtifactSizeBytes,
		)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, fmt.Errorf("MODEL_SELECTION_CHANNEL_MISSING:%s:%s", requirement.family, requirement.channel)
		}
		if err != nil {
			return nil, err
		}
		if selected.ArchitectureID != requirement.architecture {
			return nil, fmt.Errorf("MODEL_SELECTION_ARCHITECTURE_MISMATCH:%s:%s!=%s", requirement.family, selected.ArchitectureID, requirement.architecture)
		}
		if selected.CheckpointContentSHA256 != artifactContentSHA {
			return nil, fmt.Errorf("MODEL_SELECTION_CHECKPOINT_CONTENT_DRIFT:%s", requirement.family)
		}
		if requirement.channel == "PRODUCTION" && lifecycle != "PRODUCTION" {
			return nil, fmt.Errorf("MODEL_SELECTION_NOT_PRODUCTION:%s:%s", requirement.family, lifecycle)
		}
		selections = append(selections, selected)
	}
	return selections, nil
}

func BindAttemptTx(ctx context.Context, tx pgx.Tx, attemptID uuid.UUID, selections []Selection) error {
	for _, selected := range selections {
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_model_releases
			  (attempt_id,family,model_release_id,selected_channel,selected_channel_lock_version)
			VALUES ($1,$2,$3,$4,$5)
		`, attemptID, selected.Family, selected.ModelReleaseID, selected.Channel, selected.ChannelLockVersion); err != nil {
			return err
		}
	}
	return nil
}

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

func LoadAttempt(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) ([]Selection, error) {
	rows, err := pool.Query(ctx, `
		SELECT amr.family,amr.model_release_id,amr.selected_channel,amr.selected_channel_lock_version,
		       mr.architecture_id,mr.engine_release_id,mr.source_code_sha,
		       mr.checkpoint_artifact_id,mr.checkpoint_content_sha256,
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
