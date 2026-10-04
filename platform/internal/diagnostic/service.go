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

// Record is intentionally limited to observed failure identity.
//
// A failure signature is evidence of consequence, not evidence of causal owner.
// Owner attribution and repair authorization are written only by the explicit
// controlled-counterfactual attribution path.
type Record struct {
	FailureSignatureID uuid.UUID
	SignatureSHA256    string
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
		"schema":                        "RealSaS.FailureSignature.v2",
		"failing_stage_id":              failingStageID,
		"error_code":                    evidence.Code,
		"failure_class":                 evidence.Class,
		"diagnostics":                   evidence.Diagnostics,
		"reported_owner_stage_id_hint":  evidence.ReportedOwnerStageID,
		"reported_owner_hint_authority": "NON_AUTHORITATIVE",
		"causal_owner_attribution":      "NOT_PERFORMED",
		"repair_authorized":             false,
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
		return Record{
			FailureSignatureID: existingID,
			SignatureSHA256:    signatureSHA,
		}, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Record{}, err
	}

	failureID := uuid.New()
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return Record{}, err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO failure_signatures
		  (id,attempt_id,execution_id,stage_id,code,severity,signature_sha256,payload)
		VALUES ($1,$2,$3,$4,$5,'ERROR',$6,$7)
	`, failureID, attemptID, executionID, failingStageID, evidence.Code, signatureSHA, payloadJSON); err != nil {
		return Record{}, err
	}

	eventPayload, err := json.Marshal(map[string]any{
		"failure_signature_id":     failureID.String(),
		"signature_sha256":         signatureSHA,
		"failing_stage_id":         failingStageID,
		"causal_owner_attribution": "NOT_PERFORMED",
		"repair_authorized":        false,
	})
	if err != nil {
		return Record{}, err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO attempt_events(attempt_id,event_type,payload)
		VALUES ($1,'FAILURE_SIGNATURE_RECORDED',$2)
	`, attemptID, eventPayload); err != nil {
		return Record{}, err
	}

	return Record{
		FailureSignatureID: failureID,
		SignatureSHA256:    signatureSHA,
	}, nil
}
