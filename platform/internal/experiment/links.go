package experiment

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type LinkRequest struct {
	ExperimentID uuid.UUID `json:"experiment_id"`
	TargetKind   string    `json:"target_kind"`
	TargetID     uuid.UUID `json:"target_id"`
	Actor        string    `json:"actor"`
}

type LinkReceipt struct {
	ExperimentID uuid.UUID `json:"experiment_id"`
	TargetKind   string    `json:"target_kind"`
	TargetID     uuid.UUID `json:"target_id"`
	Reused       bool      `json:"reused"`
}

func Link(ctx context.Context, pool *pgxpool.Pool, req LinkRequest) (LinkReceipt, error) {
	req.TargetKind = strings.ToUpper(strings.TrimSpace(req.TargetKind))
	req.Actor = strings.TrimSpace(req.Actor)
	if req.ExperimentID == uuid.Nil || req.TargetID == uuid.Nil || req.Actor == "" {
		return LinkReceipt{}, errors.New("experiment_id/target_id/actor are required")
	}
	if req.TargetKind != "ATTEMPT" && req.TargetKind != "CAMPAIGN" {
		return LinkReceipt{}, errors.New("target_kind must be ATTEMPT or CAMPAIGN")
	}
	out := LinkReceipt{ExperimentID: req.ExperimentID, TargetKind: req.TargetKind, TargetID: req.TargetID}
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var experiment uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM experiments WHERE id=$1 FOR SHARE", req.ExperimentID).Scan(&experiment); err != nil {
			return err
		}
		var existing *uuid.UUID
		var query string
		if req.TargetKind == "ATTEMPT" {
			query = "SELECT experiment_id FROM attempts WHERE id=$1 FOR UPDATE"
		} else {
			query = "SELECT experiment_id FROM research_campaigns WHERE id=$1 FOR UPDATE"
		}
		if err := tx.QueryRow(ctx, query, req.TargetID).Scan(&existing); err != nil {
			return err
		}
		if existing != nil {
			if *existing != req.ExperimentID {
				return errors.New("EXPERIMENT_TARGET_ALREADY_LINKED")
			}
			out.Reused = true
			return nil
		}
		if req.TargetKind == "ATTEMPT" {
			if _, err := tx.Exec(ctx, "UPDATE attempts SET experiment_id=$2 WHERE id=$1", req.TargetID, req.ExperimentID); err != nil {
				return err
			}
		} else {
			if _, err := tx.Exec(ctx, "UPDATE research_campaigns SET experiment_id=$2 WHERE id=$1", req.TargetID, req.ExperimentID); err != nil {
				return err
			}
		}
		audit, _ := json.Marshal(map[string]any{"experiment_id": req.ExperimentID.String(), "target_kind": req.TargetKind, "target_id": req.TargetID.String()})
		_, err := tx.Exec(ctx, "INSERT INTO audit_events(actor,action,payload) VALUES ($1,'EXPERIMENT_TARGET_LINKED',$2)", req.Actor, audit)
		return err
	})
	return out, err
}
