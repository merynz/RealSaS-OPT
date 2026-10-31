package command

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type ProductRequalificationRequest struct {
	BaseProductRevisionID uuid.UUID
	AuthoringArtifactID   uuid.UUID
	SatisfiedCapabilities []string
	InvalidatedDomains    []string
	IdempotencyKey        string
	RequestedBy           string
}

type requalificationModelIdentity struct {
	Family                     string `json:"family"`
	ModelReleaseID             string `json:"model_release_id"`
	SelectedChannel            string `json:"selected_channel"`
	SelectedChannelLockVersion int64  `json:"selected_channel_lock_version"`
}

func SubmitProductRequalification(
	ctx context.Context,
	pool *pgxpool.Pool,
	req ProductRequalificationRequest,
) (CapabilityReceipt, error) {
	if pool == nil || req.BaseProductRevisionID == uuid.Nil || req.AuthoringArtifactID == uuid.Nil {
		return CapabilityReceipt{}, errors.New("pool, base_product_revision_id and authoring_artifact_id are required")
	}
	if len(req.SatisfiedCapabilities) == 0 || len(req.InvalidatedDomains) == 0 {
		return CapabilityReceipt{}, errors.New("satisfied_capabilities and invalidated_domains are required")
	}
	if strings.TrimSpace(req.IdempotencyKey) == "" || strings.TrimSpace(req.RequestedBy) == "" {
		return CapabilityReceipt{}, errors.New("idempotency_key and requested_by are required")
	}

	var subjectID, baseAttemptID, releaseID uuid.UUID
	var baseManifestSHA, baseAttemptState, baseAttemptKind string
	if err := pool.QueryRow(ctx, `
		SELECT pr.subject_id,pr.created_from_attempt_id,pr.manifest_sha256,
		       a.engine_release_id,a.final_state,a.kind
		FROM product_revisions pr
		JOIN attempts a ON a.id=pr.created_from_attempt_id
		WHERE pr.id=$1
	`, req.BaseProductRevisionID).Scan(
		&subjectID, &baseAttemptID, &baseManifestSHA,
		&releaseID, &baseAttemptState, &baseAttemptKind,
	); err != nil {
		return CapabilityReceipt{}, err
	}
	if releaseID == uuid.Nil || baseAttemptState != "QUALIFIED" {
		return CapabilityReceipt{}, errors.New("PRODUCT_REQUALIFICATION_QUALIFIED_BASE_REQUIRED")
	}
	if baseAttemptKind != "compile_candidate" && baseAttemptKind != "requalification_candidate" {
		return CapabilityReceipt{}, fmt.Errorf("PRODUCT_REQUALIFICATION_BASE_ATTEMPT_KIND:%s", baseAttemptKind)
	}

	var artifactType, schemaVersion, authoringSemanticSHA string
	if err := pool.QueryRow(ctx, `
		SELECT t.name,t.schema_version,a.semantic_sha256
		FROM artifacts a JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE a.id=$1
	`, req.AuthoringArtifactID).Scan(&artifactType, &schemaVersion, &authoringSemanticSHA); err != nil {
		return CapabilityReceipt{}, err
	}
	if artifactType != "RealSaS.StudioAuthoringRevision" || schemaVersion != "v1" {
		return CapabilityReceipt{}, errors.New("PRODUCT_REQUALIFICATION_STUDIO_AUTHORING_ARTIFACT_REQUIRED")
	}

	snapshot, err := capability.LoadReleaseSnapshot(ctx, pool, releaseID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	goal := capability.Goal{
		Type:            capability.GoalProductRequalification,
		Targets:         []string{capability.ProductProofCapabilityID},
		Satisfied:       append([]string(nil), req.SatisfiedCapabilities...),
		PromotionPolicy: capability.PromotionQualifiedProductOnly,
	}
	executionResolved, err := goal.Validate(snapshot.Registry)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	executionIDs := make([]string, 0, len(executionResolved))
	for _, descriptor := range executionResolved {
		if descriptor.Kind == capability.KindStage {
			return CapabilityReceipt{}, fmt.Errorf("PRODUCT_REQUALIFICATION_LEGACY_STAGE_DEPENDENCY_FORBIDDEN:%s", descriptor.ID)
		}
		executionIDs = append(executionIDs, descriptor.ID)
	}
	if len(executionIDs) == 0 || executionIDs[len(executionIDs)-1] != capability.ProductProofCapabilityID {
		return CapabilityReceipt{}, errors.New("PRODUCT_REQUALIFICATION_PROOF_TERMINAL_REQUIRED")
	}
	fullResolved, err := snapshot.Registry.Resolve(capability.ProductProofCapabilityID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	closureIDs := make([]string, 0, len(fullResolved))
	for _, descriptor := range fullResolved {
		if descriptor.Kind == capability.KindStage {
			return CapabilityReceipt{}, fmt.Errorf("PRODUCT_REQUALIFICATION_LEGACY_STAGE_DEPENDENCY_FORBIDDEN:%s", descriptor.ID)
		}
		closureIDs = append(closureIDs, descriptor.ID)
	}

	satisfied := append([]string(nil), req.SatisfiedCapabilities...)
	invalidated := append([]string(nil), req.InvalidatedDomains...)
	sort.Strings(satisfied)
	sort.Strings(invalidated)
	for i := 1; i < len(satisfied); i++ {
		if satisfied[i] == satisfied[i-1] {
			return CapabilityReceipt{}, fmt.Errorf("PRODUCT_REQUALIFICATION_SATISFIED_DUPLICATE:%s", satisfied[i])
		}
	}
	for i := 1; i < len(invalidated); i++ {
		if invalidated[i] == invalidated[i-1] {
			return CapabilityReceipt{}, fmt.Errorf("PRODUCT_REQUALIFICATION_DOMAIN_DUPLICATE:%s", invalidated[i])
		}
	}

	var releaseSHA, purpose string
	if err := pool.QueryRow(ctx, `
		SELECT release_sha256,purpose FROM engine_releases WHERE id=$1
	`, releaseID).Scan(&releaseSHA, &purpose); err != nil {
		return CapabilityReceipt{}, err
	}
	if release.Purpose(purpose) != release.PurposeProduct {
		return CapabilityReceipt{}, ErrProductReleaseRequired
	}

	modelRows, err := loadBaseRequalificationModels(ctx, pool, req.BaseProductRevisionID, baseAttemptID)
	if err != nil {
		return CapabilityReceipt{}, err
	}
	params := map[string]any{
		"schema":                       "RealSaS.ProductRequalificationParameters.v1",
		"base_product_revision_id":     req.BaseProductRevisionID.String(),
		"base_product_manifest_sha256": baseManifestSHA,
		"studio_authoring_artifact_id": req.AuthoringArtifactID.String(),
		"satisfied_capability_ids":     satisfied,
		"invalidated_domains":          invalidated,
		"execution_capability_ids":     executionIDs,
	}
	spec := map[string]any{
		"schema":                       "RealSaS.ProductRequalificationSpec.v1",
		"subject_id":                   subjectID.String(),
		"engine_release_id":            releaseID.String(),
		"engine_release_sha256":        releaseSHA,
		"capability_set_sha256":        snapshot.CapabilitySetSHA256,
		"base_product_revision_id":     req.BaseProductRevisionID.String(),
		"base_product_manifest_sha256": baseManifestSHA,
		"authoring_artifact": map[string]any{
			"id":              req.AuthoringArtifactID.String(),
			"artifact_type":   artifactType,
			"schema_version":  schemaVersion,
			"semantic_sha256": authoringSemanticSHA,
		},
		"target":                         capability.ProductProofCapabilityID,
		"authority_closure_capabilities": closureIDs,
		"execution_capabilities":         executionIDs,
		"satisfied_capabilities":         satisfied,
		"invalidated_domains":            invalidated,
		"base_model_releases":            modelRows,
		"promotion_policy":               string(capability.PromotionQualifiedProductOnly),
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
			if existing.CommandType != RunCapability || existing.SubjectID != subjectID ||
				payloadString(existing.Payload, "spec_sha256") != specSHA ||
				payloadString(existing.Payload, "engine_release_id") != releaseID.String() {
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
			out = CapabilityReceipt{CommandID: existing.ID, AttemptID: attemptID, ExecutionGoalID: goalID, ReusedIdempotencyKey: true}
			return nil
		}

		var txSubject, txBaseAttempt, txRelease uuid.UUID
		var txManifest, txFinalState string
		if err := tx.QueryRow(ctx, `
			SELECT pr.subject_id,pr.created_from_attempt_id,pr.manifest_sha256,a.engine_release_id,a.final_state
			FROM product_revisions pr JOIN attempts a ON a.id=pr.created_from_attempt_id
			WHERE pr.id=$1 FOR SHARE OF pr,a
		`, req.BaseProductRevisionID).Scan(
			&txSubject, &txBaseAttempt, &txManifest, &txRelease, &txFinalState,
		); err != nil {
			return err
		}
		if txSubject != subjectID || txBaseAttempt != baseAttemptID || txRelease != releaseID ||
			txManifest != baseManifestSHA || txFinalState != "QUALIFIED" {
			return errors.New("PRODUCT_REQUALIFICATION_BASE_REVISION_DRIFT")
		}

		for _, capabilityID := range satisfied {
			var status string
			if err := tx.QueryRow(ctx, `
				SELECT status FROM executions
				WHERE attempt_id=$1 AND stage_contract=$2
				ORDER BY retry_number DESC LIMIT 1
			`, baseAttemptID, capabilityID).Scan(&status); err != nil {
				return fmt.Errorf("PRODUCT_REQUALIFICATION_BASE_EXECUTION_REQUIRED:%s:%w", capabilityID, err)
			}
			if status != "PASS" {
				return fmt.Errorf("PRODUCT_REQUALIFICATION_BASE_EXECUTION_NOT_PASS:%s:%s", capabilityID, status)
			}
		}

		attemptID := uuid.New()
		goalID := uuid.New()
		commandID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempts(
				id,subject_id,engine_release_id,parent_attempt_id,kind,spec_sha256,created_by,final_state
			) VALUES ($1,$2,$3,$4,'requalification_candidate',$5,$6,'OPEN')
		`, attemptID, subjectID, releaseID, baseAttemptID, specSHA, req.RequestedBy); err != nil {
			return err
		}

		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_model_releases(
				attempt_id,family,model_release_id,selected_channel,selected_channel_lock_version
			)
			SELECT $1,family,model_release_id,selected_channel,selected_channel_lock_version
			FROM attempt_model_releases WHERE attempt_id=$2
		`, attemptID, baseAttemptID); err != nil {
			return err
		}

		for _, capabilityID := range satisfied {
			prefix := "capability:" + capabilityID + ":"
			tag, err := tx.Exec(ctx, `
				INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
				SELECT $1,role,artifact_id,'inherited'
				FROM attempt_artifacts
				WHERE attempt_id=$2 AND left(role,$3)=$4
			`, attemptID, baseAttemptID, len(prefix), prefix)
			if err != nil {
				return err
			}
			if tag.RowsAffected() == 0 {
				return fmt.Errorf("PRODUCT_REQUALIFICATION_BASE_ARTIFACTS_REQUIRED:%s", capabilityID)
			}
		}

		targetsJSON, _ := json.Marshal([]string{capability.ProductProofCapabilityID})
		closureJSON, _ := json.Marshal(closureIDs)
		paramsJSON, _ := json.Marshal(params)
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_goals(
				id,attempt_id,goal_type,promotion_policy,target_capability_ids,
				resolved_capability_ids,parameters,spec_sha256
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		`, goalID, attemptID, string(capability.GoalProductRequalification),
			string(capability.PromotionQualifiedProductOnly), targetsJSON, closureJSON, paramsJSON, specSHA); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_goal_inputs(execution_goal_id,ordinal,artifact_id)
			VALUES ($1,0,$2)
		`, goalID, req.AuthoringArtifactID); err != nil {
			return err
		}

		payload := map[string]any{
			"schema":                   "RealSaS.RunCapabilityCommand.v1",
			"command_id":               commandID.String(),
			"attempt_id":               attemptID.String(),
			"execution_goal_id":        goalID.String(),
			"engine_release_id":        releaseID.String(),
			"subject_id":               subjectID.String(),
			"capability_set_sha256":    snapshot.CapabilitySetSHA256,
			"spec_sha256":              specSHA,
			"product_requalification":  true,
			"base_product_revision_id": req.BaseProductRevisionID.String(),
			"authoring_artifact_id":    req.AuthoringArtifactID.String(),
		}
		if err := insertCommand(ctx, tx, commandID, RunCapability, subjectID, req.IdempotencyKey, payload); err != nil {
			return err
		}
		if err := appendOutbox(ctx, tx, "Command", commandID, "CapabilityRunRequested", map[string]any{"command_id": commandID.String()}); err != nil {
			return err
		}
		if err := appendAudit(ctx, tx, req.RequestedBy, "PRODUCT_REQUALIFICATION_ACCEPTED", subjectID, &attemptID, &req.AuthoringArtifactID, map[string]any{
			"command_id":                    commandID.String(),
			"execution_goal_id":             goalID.String(),
			"base_product_revision_id":      req.BaseProductRevisionID.String(),
			"authority_closure_capabilities": closureIDs,
			"execution_capabilities":         executionIDs,
			"satisfied_capabilities":         satisfied,
			"invalidated_domains":            invalidated,
			"promotion_policy":               string(capability.PromotionQualifiedProductOnly),
		}); err != nil {
			return err
		}
		out = CapabilityReceipt{CommandID: commandID, AttemptID: attemptID, ExecutionGoalID: goalID}
		return nil
	})
	return out, err
}

func loadBaseRequalificationModels(
	ctx context.Context,
	pool *pgxpool.Pool,
	baseRevisionID uuid.UUID,
	baseAttemptID uuid.UUID,
) ([]requalificationModelIdentity, error) {
	var drift bool
	if err := pool.QueryRow(ctx, `
		SELECT EXISTS(
			(SELECT family,model_release_id FROM attempt_model_releases WHERE attempt_id=$1
			 EXCEPT
			 SELECT family,model_release_id FROM product_revision_model_releases WHERE product_revision_id=$2)
			UNION ALL
			(SELECT family,model_release_id FROM product_revision_model_releases WHERE product_revision_id=$2
			 EXCEPT
			 SELECT family,model_release_id FROM attempt_model_releases WHERE attempt_id=$1)
		)
	`, baseAttemptID, baseRevisionID).Scan(&drift); err != nil {
		return nil, err
	}
	if drift {
		return nil, errors.New("PRODUCT_REQUALIFICATION_BASE_MODEL_LINEAGE_DRIFT")
	}
	rows, err := pool.Query(ctx, `
		SELECT family,model_release_id,selected_channel,selected_channel_lock_version
		FROM attempt_model_releases WHERE attempt_id=$1 ORDER BY family
	`, baseAttemptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []requalificationModelIdentity{}
	for rows.Next() {
		var row requalificationModelIdentity
		var modelID uuid.UUID
		if err := rows.Scan(&row.Family, &modelID, &row.SelectedChannel, &row.SelectedChannelLockVersion); err != nil {
			return nil, err
		}
		row.ModelReleaseID = modelID.String()
		out = append(out, row)
	}
	return out, rows.Err()
}
