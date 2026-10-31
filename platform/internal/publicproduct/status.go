package publicproduct

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func GetStatus(ctx context.Context, pool *pgxpool.Pool, publicID string) (Status, error) {
	session, err := loadSession(ctx, pool, publicID)
	if err != nil {
		return Status{}, err
	}
	if session.AttemptID == nil {
		return Status{
			CompileID: publicID,
			Status:    "WAITING_SOURCE_MATERIALIZATION",
			Stage:     "INPUT",
			Detail:    "Source accepted. Preparing qualified character observations.",
			Terminal:  false,
		}, nil
	}

	var finalState string
	if err := pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, *session.AttemptID).Scan(&finalState); err != nil {
		return Status{}, err
	}
	if finalState == "QUALIFIED" {
		return qualifiedProductStatus(ctx, pool, session)
	}
	if finalState == "FAILED" {
		_ = markFailed(ctx, pool, session.ID, "PRODUCT_COMPILE_FAILED")
		return Status{
			CompileID: publicID,
			Status:    "FAILED",
			Stage:     publicStage(session.CurrentCapability),
			Detail:    "Compile stopped because a product qualification gate failed.",
			Terminal:  true,
		}, nil
	}

	var stageContract, executionStatus string
	var errorCode *string
	err = pool.QueryRow(ctx, `
		SELECT stage_contract,status,error_code
		FROM executions
		WHERE attempt_id=$1
		ORDER BY started_at DESC NULLS LAST, finished_at DESC NULLS LAST
		LIMIT 1
	`, *session.AttemptID).Scan(&stageContract, &executionStatus, &errorCode)
	if errors.Is(err, pgx.ErrNoRows) {
		return Status{
			CompileID: publicID,
			Status:    "QUEUED",
			Stage:     "INPUT",
			Detail:    "Compile admitted. Waiting for the first product capability.",
			Terminal:  false,
		}, nil
	}
	if err != nil {
		return Status{}, err
	}
	stage := publicStage(&stageContract)
	if executionStatus == "FAILED" || executionStatus == "ERROR" {
		code := "PRODUCT_CAPABILITY_FAILED"
		if errorCode != nil && strings.TrimSpace(*errorCode) != "" {
			code = *errorCode
		}
		_ = markFailed(ctx, pool, session.ID, code)
		return Status{
			CompileID: publicID,
			Status:    "FAILED",
			Stage:     stage,
			Detail:    "Compile stopped because a product capability failed.",
			Terminal:  true,
		}, nil
	}
	if _, err := pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='RUNNING',current_capability=$2,error_code=NULL
		WHERE id=$1
	`, session.ID, stageContract); err != nil {
		return Status{}, err
	}
	return Status{
		CompileID: publicID,
		Status:    "RUNNING",
		Stage:     stage,
		Detail:    "Compiling qualified product state.",
		Terminal:  false,
	}, nil
}

func qualifiedProductStatus(ctx context.Context, pool *pgxpool.Pool, session sessionRecord) (Status, error) {
	var revisionID uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT id FROM product_revisions WHERE created_from_attempt_id=$1
	`, *session.AttemptID).Scan(&revisionID); err != nil {
		return Status{}, err
	}
	if session.ProductRevisionID != nil && *session.ProductRevisionID != revisionID {
		return Status{}, errors.New("PUBLIC_PRODUCT_REVISION_IDENTITY_DRIFT")
	}
	if _, err := pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='RUNNING',current_capability='workspace.materialization',
		    product_revision_id=$2,error_code=NULL
		WHERE id=$1 AND state <> 'PASS'
	`, session.ID, revisionID); err != nil {
		return Status{}, err
	}
	if err := ensureWorkspaceMaterialization(ctx, pool, session.ID, revisionID); err != nil {
		return Status{}, err
	}

	var state string
	var errorCode *string
	if err := pool.QueryRow(ctx, `
		SELECT state,last_error_code
		FROM public_product_workspace_materializations
		WHERE session_id=$1
	`, session.ID).Scan(&state, &errorCode); err != nil {
		return Status{}, err
	}
	switch state {
	case "PASS":
		if _, err := pool.Exec(ctx, `
			UPDATE public_product_compile_sessions
			SET state='PASS',current_capability='workspace.materialization',error_code=NULL
			WHERE id=$1
		`, session.ID); err != nil {
			return Status{}, err
		}
		return Status{
			CompileID:   session.PublicID,
			Status:      "PASS",
			Stage:       "FINALIZE",
			Detail:      "Product proof passed and the editable workspace is ready.",
			Terminal:    true,
			ResultReady: true,
		}, nil
	case "FAILED":
		code := "WORKSPACE_MATERIALIZATION_FAILED"
		if errorCode != nil && strings.TrimSpace(*errorCode) != "" {
			code = *errorCode
		}
		_ = markFailed(ctx, pool, session.ID, code)
		return Status{
			CompileID: session.PublicID,
			Status:    "FAILED",
			Stage:     "FINALIZE",
			Detail:    "The product qualified, but its editable workspace could not be prepared.",
			Terminal:  true,
		}, nil
	case "QUEUED", "LEASED":
		return Status{
			CompileID: session.PublicID,
			Status:    "RUNNING",
			Stage:     "FINALIZE",
			Detail:    "Product proof passed. Preparing the editable workspace.",
			Terminal:  false,
		}, nil
	default:
		return Status{}, errors.New("PUBLIC_PRODUCT_WORKSPACE_STATE_INVALID")
	}
}

func RawViewArtifactIDs(ctx context.Context, pool *pgxpool.Pool, publicID string) (map[string]uuid.UUID, error) {
	session, err := loadSession(ctx, pool, publicID)
	if err != nil {
		return nil, err
	}
	rows, err := pool.Query(ctx, `
		SELECT slot,artifact_id
		FROM public_product_compile_views
		WHERE session_id=$1 ORDER BY slot
	`, session.ID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]uuid.UUID{}
	for rows.Next() {
		var slot string
		var artifactID uuid.UUID
		if err := rows.Scan(&slot, &artifactID); err != nil {
			return nil, err
		}
		out[slot] = artifactID
	}
	return out, rows.Err()
}

func loadSession(ctx context.Context, pool *pgxpool.Pool, publicID string) (sessionRecord, error) {
	publicID = strings.TrimSpace(publicID)
	if pool == nil || publicID == "" || len(publicID) > 96 {
		return sessionRecord{}, ErrCompileNotFound
	}
	var out sessionRecord
	err := pool.QueryRow(ctx, `
		SELECT id,public_id,subject_id,engine_release_id,release_channel_lock_version,
		       source_observation_artifact_id,attempt_id,product_revision_id,state,
		       current_capability,error_code
		FROM public_product_compile_sessions
		WHERE public_id=$1
	`, publicID).Scan(
		&out.ID, &out.PublicID, &out.SubjectID, &out.EngineReleaseID, &out.ChannelLockVersion,
		&out.SourceObservationID, &out.AttemptID, &out.ProductRevisionID, &out.State,
		&out.CurrentCapability, &out.ErrorCode,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return sessionRecord{}, ErrCompileNotFound
	}
	return out, err
}

func markFailed(ctx context.Context, pool *pgxpool.Pool, sessionID uuid.UUID, code string) error {
	_, err := pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='FAILED',error_code=$2
		WHERE id=$1
	`, sessionID, code)
	return err
}

func publicStage(capability *string) string {
	if capability == nil {
		return "INPUT"
	}
	value := strings.TrimSpace(*capability)
	value = strings.TrimPrefix(value, "product.")
	if value == "" {
		return "INPUT"
	}
	return strings.ToUpper(value)
}
