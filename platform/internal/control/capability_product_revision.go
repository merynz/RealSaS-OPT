package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/product"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

const (
	modularRevisionMeshRole      = "mesh:canonical"
	modularRevisionRigRole       = "rig:canonical"
	modularRevisionSkinRole      = "skin:canonical"
	modularRevisionMechanicsRole = "mechanics:canonical"
	modularRevisionGraphicsRole  = "graphics:canonical"
	modularRevisionMotionRole    = "motion:canonical"
	modularRevisionProofRole     = "proof:product_pass"
)

type modularRevisionSource struct {
	SourceRole    string
	RevisionRole  string
	ArtifactType  string
	SchemaVersion string
}

var modularRevisionSources = []modularRevisionSource{
	{"capability:product.mesh:canonical_mesh", modularRevisionMeshRole, "RealSaS.CanonicalMeshIR", "v1"},
	{"capability:product.rig:canonical_rig", modularRevisionRigRole, "RealSaS.CanonicalRigIR", "v2"},
	{"capability:product.skin:canonical_skin", modularRevisionSkinRole, "RealSaS.CanonicalSkinIR", "v2"},
	{"capability:product.mechanics:canonical_mechanics", modularRevisionMechanicsRole, "RealSaS.CanonicalMechanicsIR", "v1"},
	{"capability:product.graphics:canonical_graphics", modularRevisionGraphicsRole, "RealSaS.CanonicalGraphicsIR", "v1"},
	{"capability:product.motion:canonical_motion", modularRevisionMotionRole, "RealSaS.CanonicalMotionIR", "v2"},
	{"capability:product.proof:product_proof", modularRevisionProofRole, "RealSaS.ProductProofBundleIR", "v1"},
}

func modularProductRevisionContract() product.Contract {
	roles := make([]product.RoleRule, 0, len(modularRevisionSources))
	for _, source := range modularRevisionSources {
		roles = append(roles, product.RoleRule{
			Role:              source.RevisionRole,
			ArtifactType:      source.ArtifactType,
			SchemaVersion:     source.SchemaVersion,
			QualificationType: "CAPABILITY_EXECUTION_PASS",
			Required:          true,
		})
	}
	return product.Contract{
		Name:    "RealSaS.ProductRevision",
		Version: "v2",
		Roles:   roles,
		Metadata: map[string]any{
			"product_graph":                    "MODULAR_CAPABILITY_V1",
			"terminal_product_pass_capability": capability.ProductProofCapabilityID,
			"legacy_stage46_required":          false,
			"legacy_runtime_package_required":  false,
			"proof_owns_product_pass_decision": true,
			"finalizer_owns_revision_seal":      true,
		},
	}
}

func loadModularRevisionBindingsTx(ctx context.Context, tx pgx.Tx, attemptID uuid.UUID) ([]revisionBinding, error) {
	bindings := make([]revisionBinding, 0, len(modularRevisionSources))
	for _, source := range modularRevisionSources {
		var binding revisionBinding
		var qualificationPassed bool
		if err := tx.QueryRow(ctx, `
			SELECT aa.role,a.id,t.name,t.schema_version,a.semantic_sha256,
			       EXISTS (
			         SELECT 1 FROM qualifications q
			         WHERE q.artifact_id=a.id
			           AND q.qualification_type='CAPABILITY_EXECUTION_PASS'
			           AND q.result='PASS'
			       )
			FROM attempt_artifacts aa
			JOIN artifacts a ON a.id=aa.artifact_id
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE aa.attempt_id=$1 AND aa.role=$2
		`, attemptID, source.SourceRole).Scan(
			&binding.Role,
			&binding.ArtifactID,
			&binding.ArtifactType,
			&binding.SchemaVersion,
			&binding.SemanticSHA256,
			&qualificationPassed,
		); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return nil, fmt.Errorf("PRODUCT_REVISION_CAPABILITY_ARTIFACT_REQUIRED:%s", source.SourceRole)
			}
			return nil, err
		}
		if binding.ArtifactType != source.ArtifactType || binding.SchemaVersion != source.SchemaVersion {
			return nil, fmt.Errorf(
				"PRODUCT_REVISION_CAPABILITY_ARTIFACT_SCHEMA_DRIFT:%s:%s:%s",
				source.SourceRole,
				binding.ArtifactType,
				binding.SchemaVersion,
			)
		}
		if !qualificationPassed {
			return nil, fmt.Errorf("PRODUCT_REVISION_CAPABILITY_QUALIFICATION_REQUIRED:%s", source.SourceRole)
		}
		binding.Role = source.RevisionRole
		bindings = append(bindings, binding)
	}
	return bindings, nil
}

func requireCapabilityProductGoalTx(ctx context.Context, tx pgx.Tx, goalID, attemptID uuid.UUID) ([]string, error) {
	var goalType, promotionPolicy string
	var targetsRaw, resolvedRaw []byte
	if err := tx.QueryRow(ctx, `
		SELECT goal_type,promotion_policy,target_capability_ids,resolved_capability_ids
		FROM execution_goals
		WHERE id=$1 AND attempt_id=$2
		FOR UPDATE
	`, goalID, attemptID).Scan(&goalType, &promotionPolicy, &targetsRaw, &resolvedRaw); err != nil {
		return nil, err
	}
	if goalType != string(capability.GoalProductCompile) || promotionPolicy != string(capability.PromotionQualifiedProductOnly) {
		return nil, errors.New("CAPABILITY_PRODUCT_FINALIZE_GOAL_POLICY_DRIFT")
	}
	var targets, resolved []string
	if err := json.Unmarshal(targetsRaw, &targets); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(resolvedRaw, &resolved); err != nil {
		return nil, err
	}
	if len(targets) != 1 || targets[0] != capability.ProductProofCapabilityID {
		return nil, errors.New("CAPABILITY_PRODUCT_FINALIZE_PROOF_TARGET_REQUIRED")
	}
	if len(resolved) == 0 || resolved[len(resolved)-1] != capability.ProductProofCapabilityID {
		return nil, errors.New("CAPABILITY_PRODUCT_FINALIZE_PROOF_TERMINAL_REQUIRED")
	}
	for _, capabilityID := range resolved {
		var status string
		if err := tx.QueryRow(ctx, `
			SELECT status FROM executions
			WHERE attempt_id=$1 AND stage_contract=$2
			ORDER BY retry_number DESC LIMIT 1
		`, attemptID, capabilityID).Scan(&status); err != nil {
			return nil, err
		}
		if status != "PASS" {
			return nil, fmt.Errorf("CAPABILITY_PRODUCT_FINALIZE_NOT_PASSED:%s:%s", capabilityID, status)
		}
	}
	return resolved, nil
}

// FinalizeCapabilityProduct mints a ProductRevision only after the terminal
// product.proof capability has passed. It does not re-evaluate product quality,
// project the modular graph back into the legacy puppet monolith, or synthesize a
// runtime package. Runtime/export materialization remains a derived post-seal step.
func (a Activities) FinalizeCapabilityProduct(ctx context.Context, request map[string]any) (orchestration.CapabilityWorkflowResult, error) {
	if a.Pool == nil {
		return orchestration.CapabilityWorkflowResult{}, errors.New("postgres pool is required")
	}
	commandRaw, ok := request["command"]
	if !ok {
		return orchestration.CapabilityWorkflowResult{}, errors.New("finalize capability product command missing")
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
	expectedReleaseID, err := uuid.Parse(command.EngineReleaseID)
	if err != nil || expectedReleaseID == uuid.Nil {
		return orchestration.CapabilityWorkflowResult{}, errors.New("CAPABILITY_PRODUCT_FINALIZE_ENGINE_RELEASE_ID_INVALID")
	}

	sealedContract, err := product.SealContract(
		ctx,
		a.Pool,
		modularProductRevisionContract(),
		"platform.finalize_capability_product.v1",
	)
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}

	var revisionID uuid.UUID
	var resolved []string
	err = persistence.WithSerializableRetry(ctx, a.Pool, 5, func(tx pgx.Tx) error {
		resolved, err = requireCapabilityProductGoalTx(ctx, tx, goalID, attemptID)
		if err != nil {
			return err
		}

		var subjectID, releaseID uuid.UUID
		var attemptKind, finalState string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,engine_release_id,kind,final_state
			FROM attempts WHERE id=$1 FOR UPDATE
		`, attemptID).Scan(&subjectID, &releaseID, &attemptKind, &finalState); err != nil {
			return err
		}
		if releaseID != expectedReleaseID {
			return errors.New("CAPABILITY_PRODUCT_FINALIZE_ENGINE_RELEASE_DRIFT")
		}
		if attemptKind != "compile_candidate" {
			return fmt.Errorf("CAPABILITY_PRODUCT_FINALIZE_ATTEMPT_KIND:%s", attemptKind)
		}
		if finalState != "OPEN" && finalState != "QUALIFIED" {
			return fmt.Errorf("CAPABILITY_PRODUCT_FINALIZE_ATTEMPT_STATE:%s", finalState)
		}
		if _, err := tx.Exec(ctx, "SELECT id FROM subjects WHERE id=$1 FOR UPDATE", subjectID); err != nil {
			return err
		}

		bindings, err := loadModularRevisionBindingsTx(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		modelRows, err := productModelIdentityTx(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		manifest := map[string]any{
			"schema":                    "RealSaS.ProductRevisionManifest.v3",
			"subject_id":                subjectID.String(),
			"engine_release_id":         releaseID.String(),
			"execution_goal_id":         goalID.String(),
			"product_contract_sha256":   sealedContract.ContractSHA256,
			"product_graph":              "MODULAR_CAPABILITY_V1",
			"terminal_proof_capability": capability.ProductProofCapabilityID,
			"bindings":                   bindings,
			"model_releases":             modelRows,
		}
		manifestSHA, err := semantic.JSONSHA256(manifest)
		if err != nil {
			return err
		}

		err = tx.QueryRow(ctx, `
			SELECT id FROM product_revisions
			WHERE subject_id=$1 AND manifest_sha256=$2
		`, subjectID, manifestSHA).Scan(&revisionID)
		if err == nil {
			if err := verifyProductRevisionModelsTx(ctx, tx, attemptID, revisionID); err != nil {
				return err
			}
			_, err = tx.Exec(ctx, "UPDATE attempts SET final_state='QUALIFIED' WHERE id=$1", attemptID)
			return err
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		var nextRevision int64
		if err := tx.QueryRow(ctx, `
			SELECT COALESCE(MAX(revision_number),0)+1
			FROM product_revisions WHERE subject_id=$1
		`, subjectID).Scan(&nextRevision); err != nil {
			return err
		}
		revisionID = uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO product_revisions
			  (id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at,product_contract_id)
			VALUES ($1,$2,$3,$4,$5,now(),$6)
		`, revisionID, subjectID, nextRevision, manifestSHA, attemptID, sealedContract.ID); err != nil {
			return err
		}
		for _, binding := range bindings {
			artifactID, err := uuid.Parse(binding.ArtifactID)
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
				INSERT INTO product_revision_artifacts(product_revision_id,role,artifact_id)
				VALUES ($1,$2,$3)
			`, revisionID, binding.Role, artifactID); err != nil {
				return err
			}
		}
		if err := bindProductRevisionModelsTx(ctx, tx, attemptID, revisionID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, "UPDATE attempts SET final_state='QUALIFIED' WHERE id=$1", attemptID); err != nil {
			return err
		}
		event, _ := json.Marshal(map[string]any{
			"product_revision_id": revisionID.String(),
			"manifest_sha256":     manifestSHA,
			"execution_goal_id":   goalID.String(),
			"product_graph":        "MODULAR_CAPABILITY_V1",
			"proof_capability_id":  capability.ProductProofCapabilityID,
			"model_release_count":  len(modelRows),
		})
		_, err = tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'PRODUCT_REVISION_SEALED',$2)
		`, attemptID, event)
		return err
	})
	if err != nil {
		return orchestration.CapabilityWorkflowResult{}, err
	}

	completed := make([]orchestration.CapabilityCommitResult, 0, len(resolved))
	for _, capabilityID := range resolved {
		completed = append(completed, orchestration.CapabilityCommitResult{CapabilityID: capabilityID, Status: "PASS"})
	}
	value := revisionID.String()
	return orchestration.CapabilityWorkflowResult{
		AttemptID:         attemptID.String(),
		Status:            "PASS",
		Completed:         completed,
		ProductRevisionID: &value,
	}, nil
}
