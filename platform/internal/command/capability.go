package command

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

const RunCapability = "RUN_CAPABILITY"

type CapabilityCompilerBridge struct {
	SubjectInputID     uuid.UUID
	CompilerRunID      string
	RunManifestPath    string
	RunLedgerPath      string
	PipelinePlanSHA256 string
}

func (b CapabilityCompilerBridge) validate() error {
	if b.SubjectInputID == uuid.Nil || b.CompilerRunID == "" || b.RunManifestPath == "" || b.RunLedgerPath == "" {
		return errors.New("CAPABILITY_COMPILER_BRIDGE_INCOMPLETE")
	}
	if len(b.PipelinePlanSHA256) != 64 {
		return errors.New("CAPABILITY_COMPILER_BRIDGE_PLAN_SHA_INVALID")
	}
	for _, ch := range b.PipelinePlanSHA256 {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
			return errors.New("CAPABILITY_COMPILER_BRIDGE_PLAN_SHA_INVALID")
		}
	}
	return nil
}

func (b CapabilityCompilerBridge) parameters(subjectInputManifestSHA string) map[string]any {
	return map[string]any{
		"schema":                        "RealSaS.CapabilityCompilerBridge.v1",
		"subject_input_id":              b.SubjectInputID.String(),
		"subject_input_manifest_sha256": subjectInputManifestSHA,
		"compiler_run_id":               b.CompilerRunID,
		"run_manifest_path":             b.RunManifestPath,
		"run_ledger_path":               b.RunLedgerPath,
		"pipeline_plan_sha256":          b.PipelinePlanSHA256,
	}
}

type CapabilityRunRequest struct {
	EngineReleaseID  uuid.UUID
	SubjectID        *uuid.UUID
	Targets          []string
	InputArtifactIDs []uuid.UUID
	Parameters       map[string]any
	CompilerBridge   *CapabilityCompilerBridge
	IdempotencyKey   string
	RequestedBy      string
}

type CapabilityReceipt struct {
	CommandID            uuid.UUID
	AttemptID            uuid.UUID
	ExecutionGoalID      uuid.UUID
	ReusedIdempotencyKey bool
}

func SubmitCapabilityRun(
	ctx context.Context,
	pool *pgxpool.Pool,
	req CapabilityRunRequest,
) (CapabilityReceipt, error) {
	if pool == nil || req.EngineReleaseID == uuid.Nil {
		return CapabilityReceipt{}, errors.New("pool and engine_release_id are required")
	}
	if len(req.Targets) == 0 || req.IdempotencyKey == "" || req.RequestedBy == "" {
		return CapabilityReceipt{}, errors.New("targets, idempotency_key and requested_by are required")
	}
	if req.Parameters == nil {
		req.Parameters = map[string]any{}
	}
	if _, reserved := req.Parameters["compiler_bridge"]; reserved {
		return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_PARAMETER_RESERVED")
	}
	if _, reserved := req.Parameters["compiler_run_id"]; reserved {
		return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_RUN_ALIAS_PARAMETER_RESERVED")
	}
	normalizedParameters := make(map[string]any, len(req.Parameters)+1)
	for key, value := range req.Parameters {
		normalizedParameters[key] = value
	}

	snapshot, err := capability.LoadReleaseSnapshot(ctx, pool, req.EngineReleaseID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	goal := capability.Goal{
		Type:            capability.GoalDeveloperRun,
		Targets:         append([]string(nil), req.Targets...),
		PromotionPolicy: capability.PromotionNever,
	}
	resolved, err := goal.Validate(snapshot.Registry)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	resolvedIDs := make([]string, 0, len(resolved))
	requiresCompilerBridge := false
	for _, descriptor := range resolved {
		resolvedIDs = append(resolvedIDs, descriptor.ID)
		if descriptor.Kind == capability.KindStage {
			requiresCompilerBridge = true
		}
	}

	bridgeManifestSHA := ""
	if requiresCompilerBridge {
		if req.CompilerBridge == nil || req.SubjectID == nil {
			return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_REQUIRED")
		}
		if err := req.CompilerBridge.validate(); err != nil {
			return CapabilityReceipt{}, err
		}
		var bridgeSubject uuid.UUID
		if err := pool.QueryRow(ctx, `
			SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1
		`, req.CompilerBridge.SubjectInputID).Scan(&bridgeSubject, &bridgeManifestSHA); err != nil {
			return CapabilityReceipt{}, err
		}
		if bridgeSubject != *req.SubjectID {
			return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_SUBJECT_INPUT_DRIFT")
		}
		_, releasedPlanSHA, err := release.LoadGraph(ctx, pool, req.EngineReleaseID)
		if err != nil {
			return CapabilityReceipt{}, err
		}
		if releasedPlanSHA != req.CompilerBridge.PipelinePlanSHA256 {
			return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_RELEASE_PLAN_DRIFT")
		}
		normalizedParameters["compiler_bridge"] = req.CompilerBridge.parameters(bridgeManifestSHA)
	} else if req.CompilerBridge != nil {
		return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_UNUSED")
	}

	type artifactIdentity struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Semantic string `json:"semantic_sha256"`
	}
	artifactRows := make([]artifactIdentity, 0, len(req.InputArtifactIDs))
	for _, id := range req.InputArtifactIDs {
		var artifactType, semanticSHA string
		if err := pool.QueryRow(ctx, `
			SELECT t.name,a.semantic_sha256
			FROM artifacts a
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE a.id=$1
		`, id).Scan(&artifactType, &semanticSHA); err != nil {
			return CapabilityReceipt{}, fmt.Errorf("developer input artifact %s: %w", id, err)
		}
		artifactRows = append(artifactRows, artifactIdentity{ID: id.String(), Type: artifactType, Semantic: semanticSHA})
	}
	sort.Slice(artifactRows, func(i, j int) bool { return artifactRows[i].ID < artifactRows[j].ID })

	var releaseSHA string
	if err := pool.QueryRow(ctx,
		"SELECT release_sha256 FROM engine_releases WHERE id=$1",
		req.EngineReleaseID,
	).Scan(&releaseSHA); err != nil {
		return CapabilityReceipt{}, err
	}

	spec := map[string]any{
		"schema":                "RealSaS.DeveloperCapabilityRunSpec.v1",
		"engine_release_id":     req.EngineReleaseID.String(),
		"engine_release_sha256": releaseSHA,
		"capability_set_sha256": snapshot.CapabilitySetSHA256,
		"targets":               req.Targets,
		"resolved_capabilities": resolvedIDs,
		"input_artifacts":       artifactRows,
		"parameters":            normalizedParameters,
		"promotion_policy":      string(capability.PromotionNever),
	}
	if req.SubjectID != nil {
		spec["subject_id"] = req.SubjectID.String()
	}
	specSHA, err := semantic.JSONSHA256(spec)
	if err != nil {
		return CapabilityReceipt{}, err
	}

	var out CapabilityReceipt
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var existingID uuid.UUID
		var existingPayload []byte
		err := tx.QueryRow(ctx, `
			SELECT id,payload FROM commands WHERE idempotency_key=$1
		`, req.IdempotencyKey).Scan(&existingID, &existingPayload)
		switch {
		case err == nil:
			var payload map[string]any
			if err := json.Unmarshal(existingPayload, &payload); err != nil {
				return err
			}
			if payloadString(payload, "spec_sha256") != specSHA {
				return ErrIdempotencyConflict
			}
			attemptID, err := payloadUUID(payload, "attempt_id")
			if err != nil {
				return err
			}
			goalID, err := payloadUUID(payload, "execution_goal_id")
			if err != nil {
				return err
			}
			out = CapabilityReceipt{CommandID: existingID, AttemptID: attemptID, ExecutionGoalID: goalID, ReusedIdempotencyKey: true}
			return nil
		case !errors.Is(err, pgx.ErrNoRows):
			return err
		}

		if req.SubjectID != nil {
			var subject uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", *req.SubjectID).Scan(&subject); err != nil {
				return err
			}
		}
		if requiresCompilerBridge {
			var bridgeSubject uuid.UUID
			var manifestSHA string
			if err := tx.QueryRow(ctx, `
				SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1 FOR SHARE
			`, req.CompilerBridge.SubjectInputID).Scan(&bridgeSubject, &manifestSHA); err != nil {
				return err
			}
			if bridgeSubject != *req.SubjectID || manifestSHA != bridgeManifestSHA {
				return errors.New("CAPABILITY_COMPILER_BRIDGE_SUBJECT_INPUT_DRIFT")
			}
		}

		attemptID := uuid.New()
		goalID := uuid.New()
		commandID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempts
			  (id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
			VALUES ($1,$2,$3,'developer',$4,$5,'OPEN')
		`, attemptID, req.SubjectID, req.EngineReleaseID, specSHA, req.RequestedBy); err != nil {
			return err
		}
		if requiresCompilerBridge {
			if _, err := tx.Exec(ctx, `
				INSERT INTO compiler_run_bindings(
					attempt_id,compiler_run_id,run_manifest_path,run_ledger_path,pipeline_plan_sha256
				) VALUES ($1,$2,$3,$4,$5)
			`, attemptID, req.CompilerBridge.CompilerRunID, req.CompilerBridge.RunManifestPath,
				req.CompilerBridge.RunLedgerPath, req.CompilerBridge.PipelinePlanSHA256); err != nil {
				return err
			}
		}
		targetsJSON, _ := json.Marshal(req.Targets)
		resolvedJSON, _ := json.Marshal(resolvedIDs)
		paramsJSON, _ := json.Marshal(normalizedParameters)
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_goals
			  (id,attempt_id,goal_type,promotion_policy,target_capability_ids,resolved_capability_ids,parameters,spec_sha256)
			VALUES ($1,$2,'DEVELOPER_RUN','NEVER',$3,$4,$5,$6)
		`, goalID, attemptID, targetsJSON, resolvedJSON, paramsJSON, specSHA); err != nil {
			return err
		}
		for ordinal, artifact := range artifactRows {
			artifactID, err := uuid.Parse(artifact.ID)
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
				INSERT INTO execution_goal_inputs(execution_goal_id,ordinal,artifact_id)
				VALUES ($1,$2,$3)
			`, goalID, ordinal, artifactID); err != nil {
				return err
			}
		}
		payload := map[string]any{
			"schema":                "RealSaS.RunCapabilityCommand.v1",
			"command_id":            commandID.String(),
			"attempt_id":            attemptID.String(),
			"execution_goal_id":     goalID.String(),
			"engine_release_id":     req.EngineReleaseID.String(),
			"capability_set_sha256": snapshot.CapabilitySetSHA256,
			"spec_sha256":           specSHA,
		}
		if req.SubjectID != nil {
			payload["subject_id"] = req.SubjectID.String()
		}
		if requiresCompilerBridge {
			payload["compiler_bridge"] = normalizedParameters["compiler_bridge"]
		}
		body, _ := json.Marshal(payload)
		if _, err := tx.Exec(ctx, `
			INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload)
			VALUES ($1,$2,$3,$4,$5)
		`, commandID, RunCapability, req.SubjectID, req.IdempotencyKey, body); err != nil {
			return err
		}
		eventBody, _ := json.Marshal(map[string]any{"command_id": commandID.String()})
		if _, err := tx.Exec(ctx, `
			INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload)
			VALUES ('Command',$1,'CapabilityRunRequested',$2)
		`, commandID, eventBody); err != nil {
			return err
		}
		auditBody, _ := json.Marshal(map[string]any{
			"command_id":            commandID.String(),
			"execution_goal_id":     goalID.String(),
			"targets":               req.Targets,
			"resolved_capabilities": resolvedIDs,
			"promotion_policy":      "NEVER",
			"compiler_bridge_used":  requiresCompilerBridge,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,attempt_id,payload)
			VALUES ($1,'DEVELOPER_CAPABILITY_RUN_ACCEPTED',$2,$3,$4)
		`, req.RequestedBy, req.SubjectID, attemptID, auditBody); err != nil {
			return err
		}
		out = CapabilityReceipt{CommandID: commandID, AttemptID: attemptID, ExecutionGoalID: goalID}
		return nil
	})
	return out, err
}
