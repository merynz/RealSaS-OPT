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

// These are the root inputs selected for this exact execution. Subject inputs
// were sealed before the Attempt. External inputs are immutable raw imported
// evidence bound to exactly (attempt, stage, role); they are copied into
// execution_artifacts so the produced stage artifact records them as semantic
// provenance inputs.
func (a Activities) boundSourceInputs(ctx context.Context, executionID uuid.UUID) ([]orchestration.ArtifactRef, error) {
	if _, err := a.Pool.Exec(ctx, `
        INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
        SELECT e.id,'input','external:' || x.role,x.artifact_id
        FROM executions e
        JOIN external_stage_evidence x
          ON x.attempt_id=e.attempt_id AND x.stage_id=e.stage_contract
        WHERE e.id=$1
        ON CONFLICT DO NOTHING
    `, executionID); err != nil {
		return nil, err
	}
	rows, err := a.Pool.Query(ctx, `SELECT ea.role,ar.id,t.name,t.schema_version,ar.semantic_sha256,
        ar.storage_key,ar.content_sha256,ar.size_bytes
        FROM execution_artifacts ea JOIN artifacts ar ON ar.id=ea.artifact_id
        JOIN artifact_types t ON t.id=ar.artifact_type_id
        WHERE ea.execution_id=$1 AND ea.relation='input'
          AND (starts_with(ea.role,'subject:') OR starts_with(ea.role,'external:'))
        ORDER BY ea.role`, executionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	refs := make([]orchestration.ArtifactRef, 0)
	for rows.Next() {
		var ref orchestration.ArtifactRef
		if err := rows.Scan(&ref.Role, &ref.ID, &ref.ArtifactType, &ref.SchemaVersion, &ref.SemanticSHA256, &ref.StorageKey, &ref.ContentSHA256, &ref.SizeBytes); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}
