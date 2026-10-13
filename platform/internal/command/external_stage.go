package command

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

const ExternalStageReady = "EXTERNAL_STAGE_READY"

type ExternalStageReadyRequest struct {
	AttemptID      uuid.UUID `json:"attempt_id"`
	StageID        string    `json:"stage_id"`
	WorkflowID     string    `json:"workflow_id"`
	EvidenceSHA    string    `json:"evidence_sha256"`
	IdempotencyKey string    `json:"idempotency_key"`
	RequestedBy    string    `json:"requested_by"`
}

func SubmitExternalStageReady(ctx context.Context, pool *pgxpool.Pool, req ExternalStageReadyRequest) (uuid.UUID, error) {
	if req.AttemptID == uuid.Nil || req.StageID == "" || req.WorkflowID == "" || req.EvidenceSHA == "" || req.IdempotencyKey == "" || req.RequestedBy == "" {
		return uuid.Nil, errors.New("external stage ready identity is incomplete")
	}
	var out uuid.UUID
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		existing, found, err := findCommandByIdempotency(ctx, tx, req.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			if existing.CommandType != ExternalStageReady ||
				payloadString(existing.Payload, "attempt_id") != req.AttemptID.String() ||
				payloadString(existing.Payload, "stage_id") != req.StageID ||
				payloadString(existing.Payload, "workflow_id") != req.WorkflowID ||
				payloadString(existing.Payload, "evidence_sha256") != req.EvidenceSHA {
				return ErrIdempotencyConflict
			}
			out = existing.ID
			return nil
		}
		var subjectID uuid.UUID
		var state string
		if err := tx.QueryRow(ctx, "SELECT subject_id,final_state FROM attempts WHERE id=$1 FOR SHARE", req.AttemptID).Scan(&subjectID, &state); err != nil {
			return err
		}
		if state != "OPEN" {
			return errors.New("EXTERNAL_STAGE_ATTEMPT_NOT_OPEN")
		}
		commandID := uuid.New()
		payload := map[string]any{
			"schema":          "RealSaS.ExternalStageReadyCommand.v1",
			"command_id":      commandID.String(),
			"attempt_id":      req.AttemptID.String(),
			"stage_id":        req.StageID,
			"workflow_id":     req.WorkflowID,
			"evidence_sha256": req.EvidenceSHA,
		}
		if err := insertCommand(ctx, tx, commandID, ExternalStageReady, subjectID, req.IdempotencyKey, payload); err != nil {
			return err
		}
		if err := appendOutbox(ctx, tx, "Command", commandID, "ExternalStageReadyRequested", map[string]any{"command_id": commandID.String()}); err != nil {
			return err
		}
		if err := appendAudit(ctx, tx, req.RequestedBy, "EXTERNAL_STAGE_READY_ACCEPTED", subjectID, &req.AttemptID, nil, map[string]any{
			"command_id": commandID.String(), "stage_id": req.StageID, "evidence_sha256": req.EvidenceSHA,
		}); err != nil {
			return err
		}
		out = commandID
		return nil
	})
	return out, err
}
