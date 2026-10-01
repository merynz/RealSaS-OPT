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
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

const RunCapability = "RUN_CAPABILITY"

type CapabilityRunRequest struct {
	EngineReleaseID  uuid.UUID
	SubjectID        *uuid.UUID
	Targets          []string
	InputArtifactIDs []uuid.UUID
	Parameters       map[string]any
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
	for _, descriptor := range resolved {
		resolvedIDs = append(resolvedIDs, descriptor.ID)
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
		"parameters":            req.Parameters,
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
		targetsJSON, _ := json.Marshal(req.Targets)
		resolvedJSON, _ := json.Marshal(resolvedIDs)
		paramsJSON, _ := json.Marshal(req.Parameters)
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_goals
			  (id,attempt_id,goal_type,promotion_policy,target_capability_ids,resolved_capability_ids,parameters,spec_sha256)
			VALUES ($1,$2,'DEVELOPER_RUN','NEVER',$3,$4,$5,$6)
		`, goalID, attemptID, targetsJSON, resolvedJSON, paramsJSON, specSHA); err != nil {
			return err
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
