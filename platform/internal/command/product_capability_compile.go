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

type ProductCapabilityCompileRequest struct {
	EngineReleaseID  uuid.UUID
	SubjectID        uuid.UUID
	InputArtifactIDs []uuid.UUID
	Parameters       map[string]any
	CompilerBridge   CapabilityCompilerBridge
	IdempotencyKey   string
	RequestedBy      string
}

type productCapabilityInputIdentity struct {
	ID            string `json:"id"`
	ArtifactType  string `json:"artifact_type"`
	SchemaVersion string `json:"schema_version"`
	SemanticSHA   string `json:"semantic_sha256"`
}

// SubmitProductCapabilityCompile creates the shipping capability-graph compile
// Attempt. The target is intentionally not caller-selectable: product.proof is
// the only terminal product-pass authority and promotion is always
// QUALIFIED_PRODUCT_ONLY. The compile_candidate insert snapshots the exact
// EngineRelease model selections through the existing database trigger.
func SubmitProductCapabilityCompile(
	ctx context.Context,
	pool *pgxpool.Pool,
	req ProductCapabilityCompileRequest,
) (CapabilityReceipt, error) {
	if pool == nil || req.EngineReleaseID == uuid.Nil || req.SubjectID == uuid.Nil {
		return CapabilityReceipt{}, errors.New("pool, engine_release_id and subject_id are required")
	}
	if req.IdempotencyKey == "" || req.RequestedBy == "" {
		return CapabilityReceipt{}, errors.New("idempotency_key and requested_by are required")
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
	if err := req.CompilerBridge.validate(); err != nil {
		return CapabilityReceipt{}, err
	}

	snapshot, err := capability.LoadReleaseSnapshot(ctx, pool, req.EngineReleaseID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	goal := capability.Goal{
		Type:            capability.GoalProductCompile,
		Targets:         []string{capability.ProductProofCapabilityID},
		PromotionPolicy: capability.PromotionQualifiedProductOnly,
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
	if !requiresCompilerBridge {
		return CapabilityReceipt{}, errors.New("PRODUCT_CAPABILITY_COMPILE_STAGE_BRIDGE_EXPECTED")
	}
	targetModelRequirements, err := capability.ResolvedProductModelRequirements(resolved)
	if err != nil {
		return CapabilityReceipt{}, err
	}

	var subjectInputSubject uuid.UUID
	var subjectInputManifestSHA string
	if err := pool.QueryRow(ctx, `
		SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1
	`, req.CompilerBridge.SubjectInputID).Scan(&subjectInputSubject, &subjectInputManifestSHA); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return CapabilityReceipt{}, ErrSubjectInputMismatch
		}
		return CapabilityReceipt{}, err
	}
	if subjectInputSubject != req.SubjectID {
		return CapabilityReceipt{}, ErrSubjectInputMismatch
	}
	_, releasedPlanSHA, err := release.LoadGraph(ctx, pool, req.EngineReleaseID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	if releasedPlanSHA != req.CompilerBridge.PipelinePlanSHA256 {
		return CapabilityReceipt{}, errors.New("CAPABILITY_COMPILER_BRIDGE_RELEASE_PLAN_DRIFT")
	}

	normalizedParameters := make(map[string]any, len(req.Parameters)+1)
	for key, value := range req.Parameters {
		normalizedParameters[key] = value
	}
	normalizedParameters["compiler_bridge"] = req.CompilerBridge.parameters(subjectInputManifestSHA)

	seenInputs := map[uuid.UUID]struct{}{}
	artifactRows := make([]productCapabilityInputIdentity, 0, len(req.InputArtifactIDs))
	for _, id := range req.InputArtifactIDs {
		if id == uuid.Nil {
			return CapabilityReceipt{}, errors.New("PRODUCT_CAPABILITY_INPUT_ARTIFACT_ID_INVALID")
		}
		if _, duplicate := seenInputs[id]; duplicate {
			return CapabilityReceipt{}, fmt.Errorf("PRODUCT_CAPABILITY_INPUT_ARTIFACT_DUPLICATE:%s", id)
		}
		seenInputs[id] = struct{}{}
		var artifactType, schemaVersion, semanticSHA string
		if err := pool.QueryRow(ctx, `
			SELECT t.name,t.schema_version,a.semantic_sha256
			FROM artifacts a
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE a.id=$1
		`, id).Scan(&artifactType, &schemaVersion, &semanticSHA); err != nil {
			return CapabilityReceipt{}, fmt.Errorf("product input artifact %s: %w", id, err)
		}
		artifactRows = append(artifactRows, productCapabilityInputIdentity{
			ID: id.String(), ArtifactType: artifactType, SchemaVersion: schemaVersion, SemanticSHA: semanticSHA,
		})
	}
	sort.Slice(artifactRows, func(i, j int) bool { return artifactRows[i].ID < artifactRows[j].ID })

	var releaseSHA, purpose string
	if err := pool.QueryRow(ctx, `
		SELECT release_sha256,purpose FROM engine_releases WHERE id=$1
	`, req.EngineReleaseID).Scan(&releaseSHA, &purpose); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return CapabilityReceipt{}, release.ErrReleaseNotFound
		}
		return CapabilityReceipt{}, err
	}
	if release.Purpose(purpose) != release.PurposeProduct {
		return CapabilityReceipt{}, ErrProductReleaseRequired
	}

	spec := map[string]any{
		"schema":                    "RealSaS.ProductCapabilityCompileSpec.v1",
		"subject_id":                req.SubjectID.String(),
		"subject_input_id":          req.CompilerBridge.SubjectInputID.String(),
		"subject_input_manifest_sha256": subjectInputManifestSHA,
		"engine_release_id":         req.EngineReleaseID.String(),
		"engine_release_sha256":     releaseSHA,
		"capability_set_sha256":     snapshot.CapabilitySetSHA256,
		"target":                    capability.ProductProofCapabilityID,
		"resolved_capabilities":     resolvedIDs,
		"target_model_requirements": targetModelRequirements,
		"input_artifacts":           artifactRows,
		"parameters":                normalizedParameters,
		"promotion_policy":          string(capability.PromotionQualifiedProductOnly),
	}
	specSHA, err := semantic.JSONSHA256(spec)
	if err != nil {
		return CapabilityReceipt{}, err
	}

	var out CapabilityReceipt
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		existing, found, err := findCommandByIdempotency(ctx, tx, req.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			if existing.CommandType != RunCapability || existing.SubjectID != req.SubjectID ||
				payloadString(existing.Payload, "spec_sha256") != specSHA ||
				payloadString(existing.Payload, "engine_release_id") != req.EngineReleaseID.String() {
				return ErrIdempotencyConflict
			}
			attemptID, err := payloadUUID(existing.Payload, "attempt_id")
			if err != nil {
				return err
			}
			goalID, err := payloadUUID(existing.Payload, "execution_goal_id")
			if err != nil {
				return err
			}
			out = CapabilityReceipt{
				CommandID: existing.ID, AttemptID: attemptID, ExecutionGoalID: goalID, ReusedIdempotencyKey: true,
			}
			return nil
		}

		var subject uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", req.SubjectID).Scan(&subject); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrSubjectNotFound
			}
			return err
		}
		var bridgeSubject uuid.UUID
		var manifestSHA string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1 FOR SHARE
		`, req.CompilerBridge.SubjectInputID).Scan(&bridgeSubject, &manifestSHA); err != nil {
			return err
		}
		if bridgeSubject != req.SubjectID || manifestSHA != subjectInputManifestSHA {
			return ErrSubjectInputMismatch
		}
		var txReleaseSHA, txPurpose string
		if err := tx.QueryRow(ctx, `
			SELECT release_sha256,purpose FROM engine_releases WHERE id=$1 FOR SHARE
		`, req.EngineReleaseID).Scan(&txReleaseSHA, &txPurpose); err != nil {
			return err
		}
		if txReleaseSHA != releaseSHA || release.Purpose(txPurpose) != release.PurposeProduct {
			return errors.New("PRODUCT_CAPABILITY_ENGINE_RELEASE_DRIFT")
		}

		attemptID := uuid.New()
		goalID := uuid.New()
		commandID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempts
			  (id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
			VALUES ($1,$2,$3,'compile_candidate',$4,$5,'OPEN')
		`, attemptID, req.SubjectID, req.EngineReleaseID, specSHA, req.RequestedBy); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO compiler_run_bindings(
				attempt_id,compiler_run_id,run_manifest_path,run_ledger_path,pipeline_plan_sha256
			) VALUES ($1,$2,$3,$4,$5)
		`, attemptID, req.CompilerBridge.CompilerRunID, req.CompilerBridge.RunManifestPath,
			req.CompilerBridge.RunLedgerPath, req.CompilerBridge.PipelinePlanSHA256); err != nil {
			return err
		}

		targetsJSON, _ := json.Marshal([]string{capability.ProductProofCapabilityID})
		resolvedJSON, _ := json.Marshal(resolvedIDs)
		paramsJSON, _ := json.Marshal(normalizedParameters)
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_goals
			  (id,attempt_id,goal_type,promotion_policy,target_capability_ids,resolved_capability_ids,parameters,spec_sha256)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		`, goalID, attemptID, string(capability.GoalProductCompile), string(capability.PromotionQualifiedProductOnly),
			targetsJSON, resolvedJSON, paramsJSON, specSHA); err != nil {
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
			"subject_id":            req.SubjectID.String(),
			"capability_set_sha256": snapshot.CapabilitySetSHA256,
			"spec_sha256":           specSHA,
			"product_compile":       true,
		}
		if err := insertCommand(ctx, tx, commandID, RunCapability, req.SubjectID, req.IdempotencyKey, payload); err != nil {
			return err
		}
		if err := appendOutbox(ctx, tx, "Command", commandID, "CapabilityRunRequested", map[string]any{"command_id": commandID.String()}); err != nil {
			return err
		}
		if err := appendAudit(ctx, tx, req.RequestedBy, "PRODUCT_CAPABILITY_COMPILE_ACCEPTED", req.SubjectID, &attemptID, nil, map[string]any{
			"command_id":                commandID.String(),
			"execution_goal_id":         goalID.String(),
			"engine_release_id":         req.EngineReleaseID.String(),
			"subject_input_id":          req.CompilerBridge.SubjectInputID.String(),
			"target":                    capability.ProductProofCapabilityID,
			"resolved_capabilities":     resolvedIDs,
			"target_model_requirements": targetModelRequirements,
			"promotion_policy":          string(capability.PromotionQualifiedProductOnly),
		}); err != nil {
			return err
		}
		out = CapabilityReceipt{CommandID: commandID, AttemptID: attemptID, ExecutionGoalID: goalID}
		return nil
	})
	return out, err
}
