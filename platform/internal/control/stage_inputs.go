package control

import (
	"context"
	"strings"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

// Send registry identities, never a guessed 'latest' run directory. All prior
// bound rows are needed because compiler adapters can read transitive ancestors.
func (a Activities) boundStageInputs(ctx context.Context, attemptID uuid.UUID) ([]orchestration.StageInput, error) {
	rows, err := a.Pool.Query(ctx, `
        SELECT aa.role,ar.id,t.name,t.schema_version,ar.semantic_sha256,
               ar.storage_key,ar.content_sha256,ar.size_bytes
        FROM attempt_artifacts aa JOIN artifacts ar ON ar.id=aa.artifact_id
        JOIN artifact_types t ON t.id=ar.artifact_type_id
        WHERE aa.attempt_id=$1 AND starts_with(aa.role,'stage:') ORDER BY aa.role
    `, attemptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	inputs := make([]orchestration.StageInput, 0)
	for rows.Next() {
		var ref orchestration.ArtifactRef
		if err := rows.Scan(&ref.Role, &ref.ID, &ref.ArtifactType, &ref.SchemaVersion, &ref.SemanticSHA256, &ref.StorageKey, &ref.ContentSHA256, &ref.SizeBytes); err != nil {
			return nil, err
		}
		inputs = append(inputs, orchestration.StageInput{StageID: strings.TrimPrefix(ref.Role, "stage:"), Artifact: ref})
	}
	return inputs, rows.Err()
}
