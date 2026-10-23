package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func (a Activities) ResolveCapabilityGoal(ctx context.Context, in orchestration.CapabilityWorkflowInput) (orchestration.ResolvedCapabilityGoal, error) {
	if a.Pool == nil {
		return orchestration.ResolvedCapabilityGoal{}, errors.New("postgres pool is required")
	}
	attemptID, err := uuid.Parse(in.AttemptID)
	if err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	goalID, err := uuid.Parse(in.ExecutionGoalID)
	if err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	releaseID, err := uuid.Parse(in.EngineReleaseID)
	if err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}

	var dbAttempt, dbRelease uuid.UUID
	var goalType, promotionPolicy, specSHA, finalState string
	var targetsRaw, resolvedRaw, paramsRaw []byte
	if err := a.Pool.QueryRow(ctx, `
		SELECT g.attempt_id,a.engine_release_id,g.goal_type,g.promotion_policy,
		       g.target_capability_ids,g.resolved_capability_ids,g.parameters,g.spec_sha256,a.final_state
		FROM execution_goals g
		JOIN attempts a ON a.id=g.attempt_id
		WHERE g.id=$1
	`, goalID).Scan(
		&dbAttempt, &dbRelease, &goalType, &promotionPolicy,
		&targetsRaw, &resolvedRaw, &paramsRaw, &specSHA, &finalState,
	); err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	if dbAttempt != attemptID || dbRelease != releaseID || specSHA != in.GoalSpecSHA256 {
		return orchestration.ResolvedCapabilityGoal{}, errors.New("CAPABILITY_GOAL_IDENTITY_DRIFT")
	}
	if finalState != "OPEN" {
		return orchestration.ResolvedCapabilityGoal{}, fmt.Errorf("CAPABILITY_ATTEMPT_NOT_OPEN:%s", finalState)
	}

	var targets, storedResolved []string
	var params map[string]any
	if err := json.Unmarshal(targetsRaw, &targets); err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	if err := json.Unmarshal(resolvedRaw, &storedResolved); err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	if err := json.Unmarshal(paramsRaw, &params); err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	snapshot, err := capability.LoadReleaseSnapshot(ctx, a.Pool, releaseID)
	if err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	resolved, err := snapshot.Registry.Resolve(targets...)
	if err != nil {
		return orchestration.ResolvedCapabilityGoal{}, err
	}
	recomputedIDs := make([]string, 0, len(resolved))
	steps := make([]orchestration.CapabilityStep, 0, len(resolved))
	for _, descriptor := range resolved {
		recomputedIDs = append(recomputedIDs, descriptor.ID)
		version, ok := snapshot.Versions[descriptor.ID]
		if !ok {
			return orchestration.ResolvedCapabilityGoal{}, fmt.Errorf("CAPABILITY_VERSION_MISSING:%s", descriptor.ID)
		}
		steps = append(steps, orchestration.CapabilityStep{
			CapabilityID:         descriptor.ID,
			Kind:                 string(descriptor.Kind),
			OwnerModuleID:        descriptor.OwnerModuleID,
			ExecutorActivity:     descriptor.ExecutorActivity,
			ImplementationSHA256: version.ImplementationSHA256,
			PolicySHA256:         version.PolicySHA256,
			ParametersSHA256:     version.ParametersSHA256,
			Dependencies:         append([]string(nil), descriptor.Dependencies...),
			Metadata:             descriptor.Metadata,
		})
	}
	if !equalStrings(storedResolved, recomputedIDs) {
		return orchestration.ResolvedCapabilityGoal{}, errors.New("CAPABILITY_GOAL_RESOLUTION_DRIFT")
	}
	return orchestration.ResolvedCapabilityGoal{
		ExecutionGoalID:     goalID.String(),
		GoalType:            goalType,
		PromotionPolicy:     promotionPolicy,
		CapabilitySetSHA256: snapshot.CapabilitySetSHA256,
		Parameters:          params,
		Steps:               steps,
	}, nil
}

func compilerBridgeParameters(params map[string]any) (uuid.UUID, string, string, string, error) {
	raw, ok := params["compiler_bridge"]
	if !ok {
		return uuid.Nil, "", "", "", errors.New("CAPABILITY_COMPILER_BRIDGE_REQUIRED")
	}
	bridge, ok := raw.(map[string]any)
	if !ok || str(bridge["schema"]) != "RealSaS.CapabilityCompilerBridge.v1" {
		return uuid.Nil, "", "", "", errors.New("CAPABILITY_COMPILER_BRIDGE_SCHEMA_DRIFT")
	}
	subjectInputID, err := uuid.Parse(str(bridge["subject_input_id"]))
	if err != nil || subjectInputID == uuid.Nil {
		return uuid.Nil, "", "", "", errors.New("CAPABILITY_COMPILER_BRIDGE_SUBJECT_INPUT_INVALID")
	}
	compilerRunID := str(bridge["compiler_run_id"])
	manifestSHA := str(bridge["subject_input_manifest_sha256"])
	planSHA := str(bridge["pipeline_plan_sha256"])
	if compilerRunID == "" || len(manifestSHA) != 64 || len(planSHA) != 64 {
		return uuid.Nil, "", "", "", errors.New("CAPABILITY_COMPILER_BRIDGE_IDENTITY_INCOMPLETE")
	}
	return subjectInputID, compilerRunID, manifestSHA, planSHA, nil
}

func str(value any) string {
	text, _ := value.(string)
	return text
}

func bindReleasedStageExecutionInputs(
	ctx context.Context,
	tx pgx.Tx,
	executionID uuid.UUID,
	subjectInputID uuid.UUID,
	attemptID uuid.UUID,
	stage stagegraph.Stage,
) error {
	rows, err := tx.Query(ctx, `
		SELECT sia.role,sia.artifact_id
		FROM subject_input_artifacts sia
		WHERE sia.subject_input_id=$1
		ORDER BY sia.ordinal
	`, subjectInputID)
	if err != nil {
		return err
	}
	type binding struct {
		role string
		id   uuid.UUID
	}
	var source []binding
	for rows.Next() {
		var item binding
		if err := rows.Scan(&item.role, &item.id); err != nil {
			rows.Close()
			return err
		}
		source = append(source, item)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()
	for _, item := range source {
		if !stage.ConsumesInput(item.role) {
			continue
		}
		role := item.role
		if !strings.HasPrefix(role, "subject:") && !strings.HasPrefix(role, "external:") {
			role = "subject:" + role
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'input',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, role, item.id); err != nil {
			return err
		}
	}
	for _, dependency := range stage.DependsOn {
		var artifactID uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT artifact_id FROM attempt_artifacts
			WHERE attempt_id=$1 AND role=$2
		`, attemptID, "stage:"+dependency).Scan(&artifactID); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return fmt.Errorf("CAPABILITY_STAGE_DEPENDENCY_ARTIFACT_REQUIRED:%s:%s", stage.ID, dependency)
			}
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'input',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, "stage:"+dependency, artifactID); err != nil {
			return err
		}
	}
	return nil
}

func executionInputRefs(ctx context.Context, tx pgx.Tx, executionID uuid.UUID) ([]orchestration.ArtifactRef, error) {
	rows, err := tx.Query(ctx, `
		SELECT ea.role,a.id,t.name,t.schema_version,a.semantic_sha256,a.storage_key,a.content_sha256,a.size_bytes
		FROM execution_artifacts ea
		JOIN artifacts a ON a.id=ea.artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE ea.execution_id=$1 AND ea.relation='input'
		ORDER BY ea.role,a.id
	`, executionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var refs []orchestration.ArtifactRef
	for rows.Next() {
		var ref orchestration.ArtifactRef
		if err := rows.Scan(
			&ref.Role, &ref.ID, &ref.ArtifactType, &ref.SchemaVersion,
			&ref.SemanticSHA256, &ref.StorageKey, &ref.ContentSHA256, &ref.SizeBytes,
		); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}

func (a Activities) PrepareCapabilityExecution(ctx context.Context, req orchestration.PrepareCapabilityExecutionRequest) (orchestration.EngineCapabilityRequest, error) {
	if a.Pool == nil {
		return orchestration.EngineCapabilityRequest{}, errors.New("postgres pool is required")
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	goalID, err := uuid.Parse(req.ExecutionGoalID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	releaseID, err := uuid.Parse(req.EngineReleaseID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	snapshot, err := capability.LoadReleaseSnapshot(ctx, a.Pool, releaseID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	descriptor, ok := snapshot.Registry.Get(req.Step.CapabilityID)
	if !ok {
		return orchestration.EngineCapabilityRequest{}, fmt.Errorf("CAPABILITY_NOT_IN_RELEASE:%s", req.Step.CapabilityID)
	}
	version := snapshot.Versions[descriptor.ID]
	if req.Step.ImplementationSHA256 != version.ImplementationSHA256 ||
		req.Step.PolicySHA256 != version.PolicySHA256 ||
		req.Step.ParametersSHA256 != version.ParametersSHA256 ||
		req.Step.ExecutorActivity != descriptor.ExecutorActivity {
		return orchestration.EngineCapabilityRequest{}, fmt.Errorf("CAPABILITY_STEP_VERSION_DRIFT:%s", descriptor.ID)
	}

	executionID := uuid.NewSHA1(uuid.NameSpaceOID, []byte("realsas:capability-execution:"+goalID.String()+":"+descriptor.ID))
	var stageParameters map[string]any
	var stageNodeSHA string
	var releasedGraph *stagegraph.Snapshot
	var pipelinePlanSHA string
	var releasedStage stagegraph.Stage
	var subjectInputID uuid.UUID
	var compilerRunID, bridgeManifestSHA, bridgePlanSHA string
	if descriptor.Kind == capability.KindStage {
		stageID, _ := descriptor.Metadata["stage_id"].(string)
		g, planSHA, err := release.LoadGraph(ctx, a.Pool, releaseID)
		if err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		graphSnapshot := g.Snapshot()
		releasedGraph, pipelinePlanSHA = &graphSnapshot, planSHA
		stage, ok := g.Get(stageID)
		if !ok {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_STAGE_DAG_DRIFT")
		}
		releasedStage = stage
		stageNodeSHA, err = stage.NodeSHA256()
		if err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		var impl, policy string
		var raw []byte
		if err := a.Pool.QueryRow(ctx, `SELECT implementation_sha256,policy_sha256,semantic_parameters
			FROM engine_release_stages WHERE release_id=$1 AND stage_id=$2`, releaseID, stageID).Scan(&impl, &policy, &raw); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if impl != version.ImplementationSHA256 || policy != version.PolicySHA256 {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_STAGE_RELEASE_DRIFT")
		}
		if err := json.Unmarshal(raw, &stageParameters); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		subjectInputID, compilerRunID, bridgeManifestSHA, bridgePlanSHA, err = compilerBridgeParameters(req.GoalParameters)
		if err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if bridgePlanSHA != pipelinePlanSHA {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_COMPILER_BRIDGE_RELEASE_PLAN_DRIFT")
		}
	}
	workflowID := "realsas:capability:" + req.CommandID

	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	defer tx.Rollback(ctx)

	var dbAttempt, dbRelease uuid.UUID
	if err := tx.QueryRow(ctx, `
		SELECT g.attempt_id,a.engine_release_id
		FROM execution_goals g JOIN attempts a ON a.id=g.attempt_id
		WHERE g.id=$1 FOR SHARE OF g,a
	`, goalID).Scan(&dbAttempt, &dbRelease); err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	if dbAttempt != attemptID || dbRelease != releaseID {
		return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_EXECUTION_GOAL_DRIFT")
	}

	if descriptor.Kind == capability.KindStage {
		var boundRunID, boundPlanSHA string
		if err := tx.QueryRow(ctx, `
			SELECT compiler_run_id,pipeline_plan_sha256 FROM compiler_run_bindings WHERE attempt_id=$1
		`, attemptID).Scan(&boundRunID, &boundPlanSHA); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if boundRunID != compilerRunID || boundPlanSHA != bridgePlanSHA {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_COMPILER_RUN_BINDING_DRIFT")
		}
		var attemptSubject, inputSubject uuid.UUID
		var manifestSHA string
		if err := tx.QueryRow(ctx, "SELECT subject_id FROM attempts WHERE id=$1", attemptID).Scan(&attemptSubject); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1
		`, subjectInputID).Scan(&inputSubject, &manifestSHA); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if inputSubject != attemptSubject || manifestSHA != bridgeManifestSHA {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_COMPILER_BRIDGE_SUBJECT_INPUT_DRIFT")
		}
	}

	tag, err := tx.Exec(ctx, `
		INSERT INTO executions
		  (id,attempt_id,workflow_id,stage_contract,status,retry_number,started_at)
		VALUES ($1,$2,$3,$4,'RUNNING',0,now())
		ON CONFLICT (id) DO NOTHING
	`, executionID, attemptID, workflowID, descriptor.ID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	if tag.RowsAffected() == 0 {
		var existingAttempt uuid.UUID
		var existingContract string
		if err := tx.QueryRow(ctx, `
			SELECT attempt_id,stage_contract FROM executions WHERE id=$1
		`, executionID).Scan(&existingAttempt, &existingContract); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		if existingAttempt != attemptID || existingContract != descriptor.ID {
			return orchestration.EngineCapabilityRequest{}, errors.New("CAPABILITY_EXECUTION_IDENTITY_COLLISION")
		}
	}

	var refs []orchestration.ArtifactRef
	if descriptor.Kind == capability.KindStage {
		if err := bindReleasedStageExecutionInputs(
			ctx, tx, executionID, subjectInputID, attemptID, releasedStage,
		); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
		refs, err = executionInputRefs(ctx, tx, executionID)
	} else {
		var inputIDs []uuid.UUID
		inputIDs, err = capabilityInputArtifactIDs(ctx, tx, goalID, attemptID, descriptor.Dependencies)
		if err == nil {
			refs, err = bindExecutionInputs(ctx, tx, executionID, inputIDs)
		}
	}
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}

	selectedModels, err := modelselection.LoadAttemptTx(ctx, tx, attemptID)
	if err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	modelRefs := modelselection.EngineRefs(selectedModels)
	if tag.RowsAffected() == 1 {
		payload, _ := json.Marshal(map[string]any{
			"execution_id":          executionID.String(),
			"capability_id":         descriptor.ID,
			"owner_module_id":       descriptor.OwnerModuleID,
			"input_artifact_count":  len(refs),
			"stage_portable_inputs": descriptor.Kind == capability.KindStage,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'CAPABILITY_EXECUTION_STARTED',$2)
		`, attemptID, payload); err != nil {
			return orchestration.EngineCapabilityRequest{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.EngineCapabilityRequest{}, err
	}
	return orchestration.EngineCapabilityRequest{
		ReleasedGraph:             releasedGraph,
		PipelinePlanSHA256:        pipelinePlanSHA,
		StageGraphNodeSHA256:      stageNodeSHA,
		StageSemanticParameters:   stageParameters,
		ExecutionID:               executionID.String(),
		CommandID:                 req.CommandID,
		AttemptID:                 req.AttemptID,
		ExecutionGoalID:           req.ExecutionGoalID,
		EngineReleaseID:           req.EngineReleaseID,
		CapabilityID:              descriptor.ID,
		Kind:                      string(descriptor.Kind),
		OwnerModuleID:             descriptor.OwnerModuleID,
		RequestedExecutorActivity: descriptor.ExecutorActivity,
		ImplementationSHA256:      version.ImplementationSHA256,
		PolicySHA256:              version.PolicySHA256,
		ParametersSHA256:          version.ParametersSHA256,
		CapabilityMetadata:        descriptor.Metadata,
		GoalParameters:            req.GoalParameters,
		InputArtifacts:            refs,
		ModelReleases:             modelRefs,
	}, nil
}

func capabilityStageSemanticSHA(req orchestration.EngineCapabilityRequest) (string, error) {
	inputs := append([]orchestration.ArtifactRef(nil), req.InputArtifacts...)
	sort.Slice(inputs, func(i, j int) bool {
		if inputs[i].Role != inputs[j].Role {
			return inputs[i].Role < inputs[j].Role
		}
		return inputs[i].SemanticSHA256 < inputs[j].SemanticSHA256
	})
	rows := make([]map[string]any, 0, len(inputs))
	for _, ref := range inputs {
		rows = append(rows, map[string]any{
			"role": ref.Role, "artifact_type": ref.ArtifactType,
			"schema_version": ref.SchemaVersion, "semantic_sha256": ref.SemanticSHA256,
		})
	}
	return semantic.JSONSHA256(map[string]any{
		"schema":                "RealSaS.CapabilityStageSemantic.v1",
		"engine_release_id":     req.EngineReleaseID,
		"capability_id":         req.CapabilityID,
		"stage_id":              str(req.CapabilityMetadata["stage_id"]),
		"implementation_sha256": req.ImplementationSHA256,
		"policy_sha256":         req.PolicySHA256,
		"parameters_sha256":     req.ParametersSHA256,
		"graph_node_sha256":     req.StageGraphNodeSHA256,
		"stage_parameters":      req.StageSemanticParameters,
		"inputs":                rows,
	})
}

func (a Activities) prepareCapabilityStageResultArtifact(
	ctx context.Context,
	req orchestration.CapabilityCommitRequest,
	expectedSemanticSHA string,
) (*preparedStageResultArtifact, orchestration.StageCommitRequest, error) {
	stageID := str(req.Request.CapabilityMetadata["stage_id"])
	if req.Request.Kind != string(capability.KindStage) || stageID == "" {
		return nil, orchestration.StageCommitRequest{}, errors.New("CAPABILITY_STAGE_RESULT_KIND_DRIFT")
	}
	outputs := append([]orchestration.EngineOutput(nil), req.Result.Outputs...)
	sort.Slice(outputs, func(i, j int) bool {
		if outputs[i].Role != outputs[j].Role {
			return outputs[i].Role < outputs[j].Role
		}
		return outputs[i].StorageKey < outputs[j].StorageKey
	})
	seen := map[string]struct{}{}
	for _, output := range outputs {
		if output.RelativePath == "" || path.IsAbs(output.RelativePath) || path.Clean(output.RelativePath) != output.RelativePath || output.RelativePath == ".." || strings.HasPrefix(output.RelativePath, "../") || output.PayloadSchema == "" {
			return nil, orchestration.StageCommitRequest{}, errors.New("CAPABILITY_STAGE_PORTABLE_OUTPUT_REQUIRED")
		}
		if output.Role == "" {
			return nil, orchestration.StageCommitRequest{}, errors.New("CAPABILITY_STAGE_OUTPUT_ROLE_REQUIRED")
		}
		if _, exists := seen[output.Role]; exists {
			return nil, orchestration.StageCommitRequest{}, fmt.Errorf("CAPABILITY_STAGE_OUTPUT_ROLE_DUPLICATE:%s", output.Role)
		}
		seen[output.Role] = struct{}{}
	}
	diagnosticsHash, err := semantic.JSONSHA256(req.Result.Diagnostics)
	if err != nil {
		return nil, orchestration.StageCommitRequest{}, err
	}
	if value := str(req.Result.Diagnostics["stage_diagnostics_hash"]); len(value) == 64 {
		diagnosticsHash = value
	}
	manifest := stageResultManifest{
		GraphNodeSHA256:        req.Request.StageGraphNodeSHA256,
		SemanticParameters:     req.Request.StageSemanticParameters,
		ImplementationSHA256:   req.Request.ImplementationSHA256,
		PolicySHA256:           req.Request.PolicySHA256,
		Schema:                 "RealSaS.StageResultManifest.v1",
		StageID:                stageID,
		CompilerStatus:         "PASS",
		ExpectedSemanticSHA256: expectedSemanticSHA,
		DiagnosticsHash:        diagnosticsHash,
		Outputs:                outputs,
	}
	raw, err := semantic.CanonicalJSON(manifest)
	if err != nil {
		return nil, orchestration.StageCommitRequest{}, err
	}
	object, err := a.Store.PutBytes(ctx, raw)
	if err != nil {
		return nil, orchestration.StageCommitRequest{}, err
	}
	params, err := json.Marshal(map[string]any{
		"stage_id": stageID, "compiler_status": "PASS", "output_count": len(outputs),
		"capability_execution": true,
	})
	if err != nil {
		return nil, orchestration.StageCommitRequest{}, err
	}
	prepared := &preparedStageResultArtifact{
		Object:                 object,
		Manifest:               manifest,
		ImplementationSHA256:   req.Request.ImplementationSHA256,
		PolicySHA256:           req.Request.PolicySHA256,
		SemanticParametersJSON: params,
	}
	stageReq := orchestration.StageCommitRequest{
		ExecutionID:            req.Request.ExecutionID,
		CommandID:              req.Request.CommandID,
		AttemptID:              req.Request.AttemptID,
		StageID:                stageID,
		ExpectedSemanticSHA256: expectedSemanticSHA,
		AllowedExecuteStageIDs: []string{stageID},
		EngineResult: orchestration.EngineStageResult{
			StageID: stageID, Status: "PASS", ExecutedStageIDs: []string{stageID},
			Outputs: outputs, DiagnosticsHash: diagnosticsHash,
		},
	}
	return prepared, stageReq, nil
}

func (a Activities) CommitCapabilityResult(ctx context.Context, req orchestration.CapabilityCommitRequest) (orchestration.CapabilityCommitResult, error) {
	if a.Pool == nil || a.Store == nil {
		return orchestration.CapabilityCommitResult{}, errors.New("postgres pool and artifact store are required")
	}
	if req.Result.CapabilityID != req.Request.CapabilityID {
		return orchestration.CapabilityCommitResult{}, errors.New("CAPABILITY_RESULT_ID_DRIFT")
	}
	executionID, err := uuid.Parse(req.Request.ExecutionID)
	if err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	attemptID, err := uuid.Parse(req.Request.AttemptID)
	if err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	produced := make([]registry.ProducedObject, 0, len(req.Result.Outputs))
	for _, output := range req.Result.Outputs {
		produced = append(produced, registry.ProducedObject{
			Role: output.Role, ArtifactType: output.ArtifactType, SchemaVersion: output.SchemaVersion,
			StorageKey: output.StorageKey, ContentSHA256: output.ContentSHA256, SizeBytes: output.SizeBytes,
			AuthorityClass: output.AuthorityClass,
		})
	}
	registrar := registry.Registrar{Store: a.Store}
	if req.Result.Status == "PASS" {
		if err := registrar.VerifyOutputs(ctx, produced); err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
	}

	var preparedStage *preparedStageResultArtifact
	var stageCommitReq orchestration.StageCommitRequest
	if req.Result.Status == "PASS" && req.Request.Kind == string(capability.KindStage) {
		semanticSHA, err := capabilityStageSemanticSHA(req.Request)
		if err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
		preparedStage, stageCommitReq, err = a.prepareCapabilityStageResultArtifact(ctx, req, semanticSHA)
		if err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
	}

	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	defer tx.Rollback(ctx)
	var dbAttempt uuid.UUID
	var contract string
	if err := tx.QueryRow(ctx, `
		SELECT attempt_id,stage_contract FROM executions WHERE id=$1 FOR UPDATE
	`, executionID).Scan(&dbAttempt, &contract); err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	if dbAttempt != attemptID || contract != req.Request.CapabilityID {
		return orchestration.CapabilityCommitResult{}, errors.New("CAPABILITY_EXECUTION_RESULT_DRIFT")
	}
	if req.Result.Status != "PASS" {
		code := "CAPABILITY_FAILED"
		class := "CAPABILITY"
		owner := req.Request.OwnerModuleID
		diagnostics := req.Result.Diagnostics
		if req.Result.Failure != nil {
			if req.Result.Failure.Code != "" {
				code = req.Result.Failure.Code
			}
			if req.Result.Failure.Class != "" {
				class = req.Result.Failure.Class
			}
			if req.Result.Failure.ReportedOwnerModuleID != "" {
				owner = req.Result.Failure.ReportedOwnerModuleID
			}
			if req.Result.Failure.Diagnostics != nil {
				diagnostics = req.Result.Failure.Diagnostics
			}
		}
		body, _ := json.Marshal(map[string]any{
			"capability_id":   req.Request.CapabilityID,
			"owner_module_id": owner,
			"class":           class,
			"diagnostics":     diagnostics,
		})
		if _, err := tx.Exec(ctx, `
			UPDATE executions SET status='FAIL',finished_at=now(),error_code=$2,error_payload=$3 WHERE id=$1
		`, executionID, code, body); err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'CAPABILITY_EXECUTION_FAILED',$2)
		`, attemptID, body); err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
		return orchestration.CapabilityCommitResult{CapabilityID: req.Request.CapabilityID, Status: "FAIL"}, nil
	}

	artifactIDs, err := registrar.RegisterCapabilityOutputs(ctx, tx, executionID, attemptID, registry.CapabilityIdentity{
		ID:                   req.Request.CapabilityID,
		ImplementationSHA256: req.Request.ImplementationSHA256,
		PolicySHA256:         req.Request.PolicySHA256,
		ParametersSHA256:     req.Request.ParametersSHA256,
	}, produced)
	if err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	if preparedStage != nil {
		stageArtifactID, err := a.commitStageResultArtifact(
			ctx, tx, executionID, attemptID, stageCommitReq, preparedStage,
		)
		if err != nil {
			return orchestration.CapabilityCommitResult{}, err
		}
		artifactIDs = append(artifactIDs, stageArtifactID)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE executions SET status='PASS',finished_at=now(),error_code=NULL,error_payload=NULL WHERE id=$1
	`, executionID); err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	artifactStrings := make([]string, 0, len(artifactIDs))
	for _, id := range artifactIDs {
		artifactStrings = append(artifactStrings, id.String())
	}
	payload, _ := json.Marshal(map[string]any{
		"execution_id":          executionID.String(),
		"capability_id":         req.Request.CapabilityID,
		"owner_module_id":       req.Request.OwnerModuleID,
		"output_artifact_ids":   artifactStrings,
		"portable_stage_result": preparedStage != nil,
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'CAPABILITY_EXECUTION_PASSED',$2)
	`, attemptID, payload); err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.CapabilityCommitResult{}, err
	}
	return orchestration.CapabilityCommitResult{
		CapabilityID: req.Request.CapabilityID, Status: "PASS", OutputArtifactIDs: artifactStrings,
	}, nil
}

func (a Activities) RecordCapabilityActivityError(ctx context.Context, req orchestration.CapabilityActivityErrorRequest) error {
	if a.Pool == nil {
		return errors.New("postgres pool is required")
	}
	executionID, err := uuid.Parse(req.ExecutionID)
	if err != nil {
		return err
	}
	attemptID, err := uuid.Parse(req.AttemptID)
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]any{
		"capability_id":   req.CapabilityID,
		"owner_module_id": req.OwnerModuleID,
		"message":         req.Error,
	})
	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `
		UPDATE executions
		SET status='FAIL',finished_at=$2,error_code='ENGINE_ACTIVITY_FAILURE',error_payload=$3
		WHERE id=$1 AND status<>'PASS'
	`, executionID, time.Now().UTC(), body); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'CAPABILITY_ACTIVITY_FAILED',$2)
	`, attemptID, body); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (a Activities) FinalizeCapabilityGoal(ctx context.Context, request map[string]any) (orchestration.CapabilityWorkflowResult, error) {
	if a.Pool == nil {
		return orchestration.CapabilityWorkflowResult{}, errors.New("postgres pool is required")
	}
	commandRaw, ok := request["command"]
	if !ok {
		return orchestration.CapabilityWorkflowResult{}, errors.New("finalize capability command missing")
	}
	body, err := json.Marshal(commandRaw)
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	var command orchestration.CapabilityWorkflowInput
	if err := json.Unmarshal(body, &command); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	attemptID, err := uuid.Parse(command.AttemptID)
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	goalID, err := uuid.Parse(command.ExecutionGoalID)
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}

	tx, err := a.Pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	defer tx.Rollback(ctx)
	var policy string
	var resolvedRaw []byte
	if err := tx.QueryRow(ctx, `
		SELECT promotion_policy,resolved_capability_ids FROM execution_goals WHERE id=$1 AND attempt_id=$2
	`, goalID, attemptID).Scan(&policy, &resolvedRaw); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	if policy != "NEVER" {
		return orchestration.CapabilityWorkflowResult{}, errors.New("CAPABILITY_FINALIZE_PROMOTION_FORBIDDEN")
	}
	var resolved []string
	if err := json.Unmarshal(resolvedRaw, &resolved); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	for _, capabilityID := range resolved {
		var status string
		if err := tx.QueryRow(ctx, `
			SELECT status FROM executions
			WHERE attempt_id=$1 AND stage_contract=$2
			ORDER BY retry_number DESC LIMIT 1
		`, attemptID, capabilityID).Scan(&status); err != nil {
			return orchestration.CapabilityWorkflowResult{}, err
		}
		if status != "PASS" {
			return orchestration.CapabilityWorkflowResult{}, fmt.Errorf("CAPABILITY_NOT_PASSED:%s:%s", capabilityID, status)
		}
	}
	if _, err := tx.Exec(ctx, `
		UPDATE attempts SET final_state='PASS' WHERE id=$1 AND final_state='OPEN'
	`, attemptID); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	payload, _ := json.Marshal(map[string]any{
		"execution_goal_id":       goalID.String(),
		"resolved_capability_ids": resolved,
		"promotion_policy":        "NEVER",
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'CAPABILITY_GOAL_COMPLETED',$2)
	`, attemptID, payload); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}

	completed := make([]orchestration.CapabilityCommitResult, 0, len(resolved))
	for _, capabilityID := range resolved {
		completed = append(completed, orchestration.CapabilityCommitResult{CapabilityID: capabilityID, Status: "PASS"})
	}
	return orchestration.CapabilityWorkflowResult{AttemptID: attemptID.String(), Status: "PASS", Completed: completed}, nil
}

func capabilityInputArtifactIDs(ctx context.Context, tx pgx.Tx, goalID, attemptID uuid.UUID, dependencies []string) ([]uuid.UUID, error) {
	seen := map[uuid.UUID]struct{}{}
	var ids []uuid.UUID
	rows, err := tx.Query(ctx, `
		SELECT artifact_id FROM execution_goal_inputs WHERE execution_goal_id=$1 ORDER BY ordinal
	`, goalID)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, err
		}
		if _, ok := seen[id]; !ok {
			seen[id] = struct{}{}
			ids = append(ids, id)
		}
	}
	rows.Close()

	for _, dependency := range dependencies {
		prefix := "capability:" + dependency + ":"
		depRows, err := tx.Query(ctx, `
			SELECT artifact_id FROM attempt_artifacts
			WHERE attempt_id=$1 AND left(role,$2)=$3
			ORDER BY role
		`, attemptID, len(prefix), prefix)
		if err != nil {
			return nil, err
		}
		for depRows.Next() {
			var id uuid.UUID
			if err := depRows.Scan(&id); err != nil {
				depRows.Close()
				return nil, err
			}
			if _, ok := seen[id]; !ok {
				seen[id] = struct{}{}
				ids = append(ids, id)
			}
		}
		depRows.Close()
	}
	return ids, nil
}

func bindExecutionInputs(ctx context.Context, tx pgx.Tx, executionID uuid.UUID, ids []uuid.UUID) ([]orchestration.ArtifactRef, error) {
	refs := make([]orchestration.ArtifactRef, 0, len(ids))
	for ordinal, id := range ids {
		var ref orchestration.ArtifactRef
		if err := tx.QueryRow(ctx, `
			SELECT a.id,t.name,t.schema_version,a.semantic_sha256,a.storage_key,a.content_sha256,a.size_bytes
			FROM artifacts a JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE a.id=$1
		`, id).Scan(
			&ref.ID, &ref.ArtifactType, &ref.SchemaVersion, &ref.SemanticSHA256,
			&ref.StorageKey, &ref.ContentSHA256, &ref.SizeBytes,
		); err != nil {
			return nil, err
		}
		ref.Role = fmt.Sprintf("input:%04d", ordinal)
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'input',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, ref.Role, id); err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].Role < refs[j].Role })
	return refs, nil
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
