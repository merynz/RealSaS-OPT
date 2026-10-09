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
	"go.temporal.io/sdk/temporal"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
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
	var attemptKind string
	if err := a.Pool.QueryRow(ctx, `
		SELECT subject_id,engine_release_id,kind
		FROM attempts
		WHERE id=$1
	`, attemptID).Scan(&dbSubject, &dbRelease, &attemptKind); err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	if dbSubject != subjectID || dbRelease != releaseID {
		return orchestration.ResolvedCompilePlan{}, errors.New("COMPILE_ATTEMPT_IDENTITY_DRIFT")
	}
	loaded, err := input.Load(ctx, a.Pool, inputID, subjectID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	mode := in.ExecutionMode
	if mode == "" {
		mode = "PRODUCT"
	}
	if (mode == "RESEARCH" && attemptKind != "research") || (mode == "PRODUCT" && attemptKind != "compile_candidate") || (mode != "PRODUCT" && mode != "RESEARCH") {
		return orchestration.ResolvedCompilePlan{}, errors.New("COMPILE_EXECUTION_LANE_DRIFT")
	}
	versions, err := release.LoadVersions(ctx, a.Pool, releaseID, mode == "PRODUCT")
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	a.Graph, _, err = release.LoadGraph(ctx, a.Pool, releaseID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	plan, err := resolver.Resolve(
		ctx,
		a.Graph,
		registry.QualifiedCatalog{Pool: a.Pool, AllowDemo: mode == "RESEARCH"},
		in.TargetStageID,
		loaded.RootInputs,
		versions,
	)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	frozen, err := attempt.LoadIntervention(ctx, a.Pool, attemptID)
	if err != nil {
		return orchestration.ResolvedCompilePlan{}, err
	}
	if frozen != nil {
		inputErr := frozen.CheckInputs(loaded)
		receipt, err := resolver.VerifyIntervention(plan, frozen.FrozenStageIDs, frozen.BaselineArtifactIDs)
		if inputErr != nil {
			err = inputErr
		}
		if err != nil {
			payload, _ := json.Marshal(map[string]any{
				"schema": "RealSaS.InterventionReuseRejection.v1", "command_id": in.CommandID,
				"target_stage_id": plan.TargetStageID, "reason": err.Error(),
			})
			if _, recordErr := a.Pool.Exec(ctx, `INSERT INTO attempt_events(attempt_id,event_type,payload)
				VALUES ($1,'INTERVENTION_REUSE_REJECTED',$2)`, attemptID, payload); recordErr != nil {
				return orchestration.ResolvedCompilePlan{}, recordErr
			}
			return orchestration.ResolvedCompilePlan{}, temporal.NewNonRetryableApplicationError(err.Error(), "InterventionContractViolation", err)
		}
		payload, err := json.Marshal(map[string]any{
			"schema": "RealSaS.InterventionReuseReceipt.v1", "command_id": in.CommandID,
			"parent_attempt_id": frozen.ParentAttemptID, "target_stage_id": plan.TargetStageID,
			"baseline_subject_input_id": frozen.BaselineSubjectInputID, "candidate_subject_input_id": loaded.SubjectInputID,
			"changed_input_roles":     frozen.Contract.ChangedInputRoles,
			"input_changed_stage_ids": frozen.InputChangedStageIDs,
			"receipt":                 receipt,
		})
		if err != nil {
			return orchestration.ResolvedCompilePlan{}, err
		}
		if _, err := a.Pool.Exec(ctx, `INSERT INTO attempt_events(attempt_id,event_type,payload)
			SELECT $1,'INTERVENTION_REUSE_VERIFIED',$2::jsonb WHERE NOT EXISTS (
			 SELECT 1 FROM attempt_events WHERE attempt_id=$1 AND event_type='INTERVENTION_REUSE_VERIFIED' AND payload=$2::jsonb
			)`, attemptID, payload); err != nil {
			return orchestration.ResolvedCompilePlan{}, err
		}
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
	var releasedPlanSHA string
	a.Graph, releasedPlanSHA, err = release.LoadGraph(ctx, a.Pool, releaseID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	if !slices.Contains(req.AllowedExecuteStageIDs, req.StageID) {
		return orchestration.EngineStageRequest{}, errors.New("EXECUTION_STAGE_NOT_IN_ALLOWED_SCOPE")
	}
	if _, ok := a.Graph.Get(req.StageID); !ok {
		return orchestration.EngineStageRequest{}, fmt.Errorf("unknown stage %s", req.StageID)
	}
	frozen, err := attempt.LoadIntervention(ctx, a.Pool, attemptID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	if err := frozen.CheckExecution(req.StageID); err != nil {
		return orchestration.EngineStageRequest{}, temporal.NewNonRetryableApplicationError(err.Error(), "InterventionContractViolation", err)
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
	if releasedPlanSHA != "" && releasedPlanSHA != planSHA {
		return orchestration.EngineStageRequest{}, errors.New("EXECUTION_RELEASE_PIPELINE_PLAN_DRIFT")
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
	if err := a.bindStageExecutionInputs(ctx, tx, executionID, req.SubjectInputID, attemptID, req.StageID); err != nil {
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
	inputs, err := a.boundStageInputs(ctx, attemptID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	sources, err := a.boundSourceInputs(ctx, executionID)
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	var implementationSHA, policySHA, kind string
	var parametersRaw []byte
	if err := a.Pool.QueryRow(ctx, `SELECT ers.implementation_sha256,ers.policy_sha256,a.kind,ers.semantic_parameters FROM attempts a JOIN engine_release_stages ers ON ers.release_id=a.engine_release_id WHERE a.id=$1 AND ers.stage_id=$2`, attemptID, req.StageID).Scan(&implementationSHA, &policySHA, &kind, &parametersRaw); err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	var parameters map[string]any
	if err := json.Unmarshal(parametersRaw, &parameters); err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	mode := "PRODUCT"
	if kind == "research" {
		mode = "RESEARCH"
	}
	stage, _ := a.Graph.Get(req.StageID)
	nodeSHA, err := stage.NodeSHA256()
	if err != nil {
		return orchestration.EngineStageRequest{}, err
	}
	return orchestration.EngineStageRequest{
		ReleasedGraph:      a.Graph.Snapshot(),
		SourceInputs:       sources,
		GraphNodeSHA256:    nodeSHA,
		SemanticParameters: parameters,
		ExecutionMode:      mode, ImplementationSHA256: implementationSHA, PolicySHA256: policySHA, InputStages: inputs,
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
	frozen, err := attempt.LoadIntervention(ctx, a.Pool, attemptID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if err := frozen.CheckReuse(req.StageID, artifactID); err != nil {
		return orchestration.StageCommitResult{}, temporal.NewNonRetryableApplicationError(err.Error(), "InterventionContractViolation", err)
	}
	var semanticSHA string
	var qualified bool
	var kind string
	if err := a.Pool.QueryRow(ctx, "SELECT kind FROM attempts WHERE id=$1", attemptID).Scan(&kind); err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if err := a.Pool.QueryRow(ctx, `
		SELECT a.semantic_sha256,
		       EXISTS (
		         SELECT 1 FROM qualifications q
		         WHERE q.artifact_id=a.id
		           AND (q.qualification_type='REUSE_ELIGIBLE' OR ($2='research' AND q.qualification_type='DEMO_REUSE_ELIGIBLE'))
		           AND q.result='PASS'
		       )
		FROM artifacts a
		WHERE a.id=$1
	`, artifactID, kind).Scan(&semanticSHA, &qualified); err != nil {
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
	frozen, err := attempt.LoadIntervention(ctx, a.Pool, attemptID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if err := frozen.CheckExecution(req.StageID); err != nil {
		return orchestration.StageCommitResult{}, temporal.NewNonRetryableApplicationError(err.Error(), "InterventionContractViolation", err)
	}
	a.Graph, err = a.attemptGraph(ctx, attemptID)
	if err != nil {
		return orchestration.StageCommitResult{}, err
	}
	if req.EngineResult.StageID != req.StageID {
		return orchestration.StageCommitResult{}, errors.New("ENGINE_STAGE_RESULT_ID_DRIFT")
	}
	for _, stageID := range req.EngineResult.ExecutedStageIDs {
		if err := frozen.CheckExecution(stageID); err != nil {
			return orchestration.StageCommitResult{}, temporal.NewNonRetryableApplicationError(err.Error(), "InterventionContractViolation", err)
		}
		if !slices.Contains(req.AllowedExecuteStageIDs, stageID) {
			return orchestration.StageCommitResult{}, fmt.Errorf("COMPILER_PLATFORM_PLAN_DRIFT:%s", stageID)
		}
	}

	success := req.EngineResult.Status == "PASS" || req.EngineResult.Status == "PASS_DEMO_ONLY"
	var prepared *preparedStageResultArtifact
	if success {
		prepared, err = a.prepareStageResultArtifact(ctx, attemptID, req)
		if err != nil {
			return orchestration.StageCommitResult{}, err
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

	artifactID, err := a.commitStageResultArtifact(ctx, tx, executionID, attemptID, req, prepared)
	if err != nil {
		return orchestration.StageCommitResult{}, err
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
	value := artifactID.String()
	return orchestration.StageCommitResult{StageID: req.StageID, Status: "PASS", ArtifactID: &value}, nil
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
	a.Graph, err = a.attemptGraph(ctx, attemptID)
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
