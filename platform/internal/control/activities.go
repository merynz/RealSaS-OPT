package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/diagnostic"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/resolver"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type Activities struct {
	Pool  *pgxpool.Pool
	Graph *stagegraph.Graph
	Store artifactstore.Store
}

func (a Activities) Validate() error {
	if a.Pool == nil || a.Graph == nil {
		return errors.New("control activities require postgres pool and stage graph")
	}
	return nil
}

func (a Activities) ResolveCompilePlan(ctx context.Context, in orchestration.CompileWorkflowInput) (orchestration.ResolvedCompilePlan, error) {
	if err := a.Validate(); err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	attemptID, err := uuid.Parse(in.AttemptID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	subjectID, err := uuid.Parse(in.SubjectID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	releaseID, err := uuid.Parse(in.EngineReleaseID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	inputID, err := uuid.Parse(in.SubjectInputID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}

	var dbSubject, dbRelease uuid.UUID
	if err := a.Pool.QueryRow(ctx, `
		SELECT subject_id,engine_release_id
		FROM attempts
		WHERE id=$1
	`, attemptID).Scan(&dbSubject, &dbRelease); err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	if dbSubject != subjectID || dbRelease != releaseID {
		return orchestration.ResolvedCompilePlan{}, errors.New("COMPILE_ATTEMPT_IDENTITY_DRIFT")
	}
	loaded, err := input.Load(ctx, a.Pool, inputID, subjectID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	versions, err := release.LoadVersions(ctx, a.Pool, releaseID, true)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	plan, err := resolver.Resolve(
		ctx,
		a.Graph,
		registry.QualifiedCatalog{Pool: a.Pool},
		in.TargetStageID,
		loaded.RootInputs,
		versions,
	)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	out := orchestration.ResolvedCompilePlan{
		TargetStageID: plan.TargetStageID,
		Stages:        make([]orchestration.ResolvedStage, 0, len(plan.Stages)),
	}
	for _, row := range plan.Stages {
		var artifactID *string
		if row.ReusableArtifactID != nil {
			value := row.ReusableArtifactID.String()
			artifactID = &value
		}
		out.Stages = append(out.Stages, orchestration.ResolvedStage{
			StageID:                row.StageID,
			Action:                 string(row.Action),
			ExpectedSemanticSHA256: row.ExpectedSemanticSHA256,
			ReusableArtifactID:     artifactID,
			Reason:                 row.Reason,
		})
	}
	return out, nil
}

func (a Activities) PrepareStageExecution(ctx context.Context, req orchestration.PrepareStageExecutionRequest) (orchestration.EngineStageRequest, error) {
	if err := a.Validate(); err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	subjectID, err := uuid.Parse(req.SubjectID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	releaseID, err := uuid.Parse(req.EngineReleaseID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	if !slices.Contains(req.AllowedExecuteStageIDs, req.StageID) {
		return orchestration.EngineStageRequest{}, errors.New("EXECUTION_STAGE_NOT_IN_ALLOWED_SCOPE")
	}
	if _, ok := a.Graph.Get(req.StageID); !ok {
		return orchestration.EngineStageRequest{}, fmt.Errorf("unknown stage %s", req.StageID)
	}

	var dbSubject, dbRelease uuid.UUID
	var compilerRunID, planSHA string
	if err := a.Pool.QueryRow(ctx, `
		SELECT a.subject_id,a.engine_release_id,b.compiler_run_id,b.pipeline_plan_sha256
		FROM attempts a
		JOIN compiler_run_bindings b ON b.attempt_id=a.id
		WHERE a.id=$1
	`, attemptID).Scan(&dbSubject, &dbRelease, &compilerRunID, &planSHA); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return orchestration.EngineStageRequest{}, errors.New("COMPILER_RUN_BINDING_REQUIRED")
		}
		return orchestration.EngineStageRequest{}, err
	}
	if dbSubject != subjectID || dbRelease != releaseID {
		return orchestration.EngineStageRequest{}, errors.New("EXECUTION_ATTEMPT_IDENTITY_DRIFT")
	}

	executionID := uuid.NewSHA1(uuid.NameSpaceOID, []byte("realsas:execution:"+req.CommandID+":"+req.StageID))
	workflowID := "realsas:compile:" + req.CommandID
	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	defer tx.Rollback(ctx)

	tag, err := tx.Exec(ctx, `
		INSERT INTO executions
		  (id,attempt_id,workflow_id,stage_contract,status,retry_number,started_at)
		VALUES ($1,$2,$3,$4,'RUNNING',0,now())
		ON CONFLICT (id) DO NOTHING
	`, executionID, attemptID, workflowID, req.StageID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	if tag.RowsAffected() == 1 {
		payload, _ := json.Marshal(map[string]any{
			"execution_id": executionID.String(),
			"stage_id":     req.StageID,
			"retry_number": 0,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'STAGE_EXECUTION_STARTED',$2)
		`, attemptID, payload); err != nil {
			return orchestration.EngineStageRequest{}, err
		}
	} else {
		var existingAttempt uuid.UUID
		var existingStage, status string
		if err := tx.QueryRow(ctx, `
			SELECT attempt_id,stage_contract,status FROM executions WHERE id=$1
		`, executionID).Scan(&existingAttempt, &existingStage, &status); err != nil {
			return orchestration.EngineStageRequest{}, err
		}
		if existingAttempt != attemptID || existingStage != req.StageID {
			return orchestration.EngineStageRequest{}, errors.New("EXECUTION_IDENTITY_COLLISION")
		}
		if status == "PASS" {
			return orchestration.EngineStageRequest{}, errors.New("EXECUTION_ALREADY_PASSED")
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	return orchestration.EngineStageRequest{
		ExecutionID:            executionID.String(),
		CommandID:              req.CommandID,
		AttemptID:              req.AttemptID,
		SubjectID:              req.SubjectID,
		EngineReleaseID:        req.EngineReleaseID,
		CompilerRunID:          compilerRunID,
		PipelinePlanSHA256:     planSHA,
		StageID:                req.StageID,
		ExpectedSemanticSHA256: req.ExpectedSemanticSHA256,
		AllowedExecuteStageIDs: append([]string(nil), req.AllowedExecuteStageIDs...),
	}, nil
}

func (a Activities) BindReusedStage(ctx context.Context, req orchestration.BindReusedStageRequest) (orchestration.StageCommitResult, error) {
	if err := a.Validate(); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	artifactID, err := uuid.Parse(req.ArtifactID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	var semanticSHA string
	var qualified bool
	if err := a.Pool.QueryRow(ctx, `
		SELECT a.semantic_sha256,
		       EXISTS (
		         SELECT 1 FROM qualifications q
		         WHERE q.artifact_id=a.id
		           AND q.qualification_type='REUSE_ELIGIBLE'
		           AND q.result='PASS'
		       )
		FROM artifacts a
		WHERE a.id=$1
	`, artifactID).Scan(&semanticSHA, &qualified); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if semanticSHA != req.ExpectedSemanticSHA256 || !qualified {
		return orchestration.StageCommitResult{}, errors.New("REUSE_ARTIFACT_IDENTITY_OR_QUALIFICATION_DRIFT")
	}

	tx, err := a.Pool.Begin(ctx)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	defer tx.Rollback(ctx)
	role := "stage:" + req.StageID
	tag, err := tx.Exec(ctx, `
		INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
		VALUES ($1,$2,$3,'inherited')
		ON CONFLICT (attempt_id,role) DO NOTHING
	`, attemptID, role, artifactID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if tag.RowsAffected() == 0 {
		var existing uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT artifact_id FROM attempt_artifacts WHERE attempt_id=$1 AND role=$2
		`, attemptID, role).Scan(&existing); err != nil {
			return orchestration.StageCommitResult{}, err
		}
		if existing != artifactID {
			return orchestration.StageCommitResult{}, errors.New("ATTEMPT_REUSE_ARTIFACT_DRIFT")
		}
	} else {
		payload, _ := json.Marshal(map[string]any{"stage_id": req.StageID, "artifact_id": artifactID.String()})
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'STAGE_REUSED',$2)
		`, attemptID, payload); err != nil {
			return orchestration.StageCommitResult{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	value := artifactID.String()
	return orchestration.StageCommitResult{StageID: req.StageID, Status: "REUSED", ArtifactID: &value}, nil
}

func (a Activities) CommitStageResult(ctx context.Context, req orchestration.StageCommitRequest) (orchestration.StageCommitResult, error) {
	if err := a.Validate(); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	executionID, err := uuid.Parse(req.ExecutionID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if req.EngineResult.StageID != req.StageID {
		return orchestration.StageCommitResult{}, errors.New("ENGINE_STAGE_RESULT_ID_DRIFT")
	}
	for _, stageID := range req.EngineResult.ExecutedStageIDs {
		if !slices.Contains(req.AllowedExecuteStageIDs, stageID) {
			return orchestration.StageCommitResult{}, fmt.Errorf("COMPILER_PLATFORM_PLAN_DRIFT:%s", stageID)
		}
	}

	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	defer tx.Rollback(ctx)
	var dbAttempt uuid.UUID
	var dbStage string
	if err := tx.QueryRow(ctx, `
		SELECT attempt_id,stage_contract FROM executions WHERE id=$1 FOR UPDATE
	`, executionID).Scan(&dbAttempt, &dbStage); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if dbAttempt != attemptID || dbStage != req.StageID {
		return orchestration.StageCommitResult{}, errors.New("EXECUTION_RESULT_IDENTITY_DRIFT")
	}

	success := req.EngineResult.Status == "PASS" || req.EngineResult.Status == "PASS_DEMO_ONLY"
	if !success {
		evidence := orchestration.EngineFailureEvidence{
			Code:        "COMPILER_STAGE_FAILED",
			Class:       "STAGE",
			Diagnostics: map[string]any{},
		}
		if req.EngineResult.Failure != nil {
			evidence = *req.EngineResult.Failure
		}
		errorPayload, _ := json.Marshal(map[string]any{
			"compiler_status":    req.EngineResult.Status,
			"executed_stage_ids": req.EngineResult.ExecutedStageIDs,
			"diagnostics_hash":   req.EngineResult.DiagnosticsHash,
			"failure":            evidence,
		})
		if _, err := tx.Exec(ctx, `
			UPDATE executions
			SET status='FAIL',finished_at=now(),error_code=$2,error_payload=$3
			WHERE id=$1
		`, executionID, evidence.Code, errorPayload); err != nil {
			return orchestration.StageCommitResult{}, err
		}
		if _, err := diagnostic.RecordFailure(ctx, tx, a.Graph, attemptID, &executionID, req.StageID, evidence); err != nil {
			return orchestration.StageCommitResult{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return orchestration.StageCommitResult{}, err
		}
		return orchestration.StageCommitResult{StageID: req.StageID, Status: "FAIL"}, nil
	}

	if _, err := tx.Exec(ctx, `
		UPDATE executions
		SET status='PASS',finished_at=now(),error_code=NULL,error_payload=NULL
		WHERE id=$1
	`, executionID); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	payload, _ := json.Marshal(map[string]any{
		"execution_id":       executionID.String(),
		"stage_id":           req.StageID,
		"compiler_status":    req.EngineResult.Status,
		"executed_stage_ids": req.EngineResult.ExecutedStageIDs,
		"output_count":       len(req.EngineResult.Outputs),
		"diagnostics_hash":   req.EngineResult.DiagnosticsHash,
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'STAGE_EXECUTION_PASSED',$2)
	`, attemptID, payload); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	return orchestration.StageCommitResult{StageID: req.StageID, Status: "PASS"}, nil
}

func (a Activities) RecordStageActivityError(ctx context.Context, req orchestration.StageActivityErrorRequest) error {
	if err := a.Validate(); err != nil {
		return err
	}
	executionID, err := uuid.Parse(req.ExecutionID)
	if err != nil {
		return err
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return err
	}
	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	var currentStatus string
	if err := tx.QueryRow(ctx, "SELECT status FROM executions WHERE id=$1 FOR UPDATE", executionID).Scan(&currentStatus); err != nil {
		return err
	}
	if currentStatus == "PASS" {
		return errors.New("CANNOT_FAIL_PASSED_EXECUTION")
	}
	body, _ := json.Marshal(map[string]any{"activity_error": req.Error})
	if _, err := tx.Exec(ctx, `
		UPDATE executions
		SET status='FAIL',finished_at=$2,error_code='ENGINE_ACTIVITY_FAILURE',error_payload=$3
		WHERE id=$1
	`, executionID, time.Now().UTC(), body); err != nil {
		return err
	}
	_, err = diagnostic.RecordFailure(ctx, tx, a.Graph, attemptID, &executionID, req.StageID, orchestration.EngineFailureEvidence{
		Code:        "ENGINE_ACTIVITY_FAILURE",
		Class:       "INFRA",
		Diagnostics: map[string]any{"message": req.Error},
	})
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}
