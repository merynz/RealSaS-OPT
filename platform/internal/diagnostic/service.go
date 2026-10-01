package diagnostic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type Record struct {
	FailureSignatureID  uuid.UUID
	OwnerAttributionID  uuid.UUID
	RepairDirectiveID   uuid.UUID
	OwnerStageID        string
	InvalidatedStageIDs []string
	SignatureSHA256     string
}

func RecordFailure(
	ctx context.Context,
	tx pgx.Tx,
	graph *stagegraph.Graph,
	attemptID uuid.UUID,
	executionID *uuid.UUID,
	failingStageID string,
	evidence orchestration.EngineFailureEvidence,
) (Record, error) {
	if _, ok := graph.Get(failingStageID); !ok {
		return Record{}, fmt.Errorf("unknown failing stage %s", failingStageID)
	}
	owner := evidence.ReportedOwnerStageID
	if owner == "" {
		owner = failingStageID
	}
	if _, ok := graph.Get(owner); !ok {
		return Record{}, fmt.Errorf("unknown failure owner %s", owner)
	}
	ancestors, err := graph.AncestorsIncluding(failingStageID)
	if err != nil {
		return Record{}, err
	}
	legal := false
	for _, stageID := range ancestors {
		if stageID == owner {
			legal = true
			break
		}
	}
	if !legal {
		return Record{}, fmt.Errorf("ILLEGAL_FAILURE_OWNER:%s:%s", failingStageID, owner)
	}
	invalidated, err := graph.DescendantsIncluding(owner)
	if err != nil {
		return Record{}, err
	}
	if evidence.Code == "" {
		evidence.Code = "COMPILER_STAGE_FAILED"
	}
	if evidence.Class == "" {
		evidence.Class = "STAGE"
	}
	if evidence.Diagnostics == nil {
		evidence.Diagnostics = map[string]any{}
	}
	payload := map[string]any{
		"schema":           "RealSaS.FailureSignature.v1",
		"failing_stage_id": failingStageID,
		"error_code":       evidence.Code,
		"failure_class":    evidence.Class,
		"diagnostics":      evidence.Diagnostics,
	}
	signatureSHA, err := semantic.JSONSHA256(payload)
	if err != nil {
		return Record{}, err
	}

	var existingID uuid.UUID
	err = tx.QueryRow(ctx, `
		SELECT id FROM failure_signatures
		WHERE attempt_id=$1 AND signature_sha256=$2
	`, attemptID, signatureSHA).Scan(&existingID)
	if err == nil {
		var attributionID, directiveID uuid.UUID
		var existingOwner string
		var invalidatedJSON []byte
		if err := tx.QueryRow(ctx, `
			SELECT oa.id,oa.owner_stage_id,rd.id,rd.invalidated_stage_ids
			FROM owner_attributions oa
			JOIN repair_directives rd ON rd.failure_signature_id=oa.failure_signature_id
			WHERE oa.failure_signature_id=$1 AND rd.status='OPEN'
			ORDER BY rd.created_at DESC
			LIMIT 1
		`, existingID).Scan(&attributionID, &existingOwner, &directiveID, &invalidatedJSON); err != nil {
			return Record{}, err
		}
		var body struct {
			StageIDs []string `json:"stage_ids"`
		}
		if err := json.Unmarshal(invalidatedJSON, &body); err != nil {
			return Record{}, err
		}
		return Record{existingID, attributionID, directiveID, existingOwner, body.StageIDs, signatureSHA}, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Record{}, err
	}

	failureID := uuid.New()
	attributionID := uuid.New()
	directiveID := uuid.New()
	payloadJSON, _ := json.Marshal(payload)
	if _, err := tx.Exec(ctx, `
		INSERT INTO failure_signatures
		  (id,attempt_id,execution_id,stage_id,code,severity,signature_sha256,payload)
		VALUES ($1,$2,$3,$4,$5,'ERROR',$6,$7)
	`, failureID, attemptID, executionID, failingStageID, evidence.Code, signatureSHA, payloadJSON); err != nil {
		return Record{}, err
	}
	ownerKind := "STAGE"
	if evidence.Class == "INPUT" || evidence.Class == "INFRA" {
		ownerKind = evidence.Class
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO owner_attributions
		  (id,failure_signature_id,owner_stage_id,owner_kind,reason)
		VALUES ($1,$2,$3,$4,$5)
	`, attributionID, failureID, owner, ownerKind,
		fmt.Sprintf("owner %s is within dependency ancestry of failing stage %s", owner, failingStageID)); err != nil {
		return Record{}, err
	}
	directiveType := "REEXECUTE_OWNER_STAGE"
	switch evidence.Class {
	case "INPUT":
		directiveType = "CHANGE_INPUT"
	case "INFRA":
		directiveType = "RETRY_INFRA"
	case "POLICY":
		directiveType = "CHANGE_POLICY"
	case "CODE":
		directiveType = "CHANGE_IMPLEMENTATION"
	case "ARTIFACT":
		directiveType = "REBUILD_OWNER_STAGE"
	}
	invalidatedJSON, _ := json.Marshal(map[string]any{"stage_ids": invalidated})
	directivePayload, _ := json.Marshal(map[string]any{
		"schema":           "RealSaS.RepairDirective.v1",
		"owner_stage_id":   owner,
		"failing_stage_id": failingStageID,
		"error_code":       evidence.Code,
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO repair_directives
		  (id,failure_signature_id,owner_stage_id,directive_type,invalidated_stage_ids,payload,status)
		VALUES ($1,$2,$3,$4,$5,$6,'OPEN')
	`, directiveID, failureID, owner, directiveType, invalidatedJSON, directivePayload); err != nil {
		return Record{}, err
	}
	eventPayload, _ := json.Marshal(map[string]any{
		"failure_signature_id":  failureID.String(),
		"owner_stage_id":        owner,
		"repair_directive_id":   directiveID.String(),
		"invalidated_stage_ids": invalidated,
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'FAILURE_LOCALIZED',$2)
	`, attemptID, eventPayload); err != nil {
		return Record{}, err
	}
	return Record{failureID, attributionID, directiveID, owner, invalidated, signatureSHA}, nil
}
