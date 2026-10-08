package dispatch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/api/workflowservice/v1"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type WorkflowDescriber interface {
	DescribeWorkflowExecution(context.Context, string, string) (*workflowservice.DescribeWorkflowExecutionResponse, error)
}

func unsuccessfulTerminal(status enumspb.WorkflowExecutionStatus) bool {
	switch status {
	case enumspb.WORKFLOW_EXECUTION_STATUS_FAILED,
		enumspb.WORKFLOW_EXECUTION_STATUS_CANCELED,
		enumspb.WORKFLOW_EXECUTION_STATUS_TERMINATED,
		enumspb.WORKFLOW_EXECUTION_STATUS_TIMED_OUT:
		return true
	default:
		return false
	}
}

// ReconcileFailedAttempts repairs lifecycle bookkeeping only. Temporal is the
// durable execution witness; failure artifacts and diagnostics remain intact.
// Running, unknown, undelivered and successfully completed workflows cannot be
// failed here. In particular, this path never grants scientific qualification.
func ReconcileFailedAttempts(ctx context.Context, pool *pgxpool.Pool, temporalClient WorkflowDescriber, limit int) (int, error) {
	if pool == nil || temporalClient == nil || limit < 1 {
		return 0, errors.New("reconciliation requires database, Temporal and positive limit")
	}
	rows, err := pool.Query(ctx, `
		SELECT a.id,c.id,c.command_type,c.payload
		FROM attempts a JOIN commands c ON c.payload->>'attempt_id'=a.id::text
		WHERE a.final_state='OPEN'
		  AND c.command_type IN ('COMPILE_SUBJECT','RUN_CAPABILITY')
		  AND (SELECT count(*) FROM commands x WHERE x.payload->>'attempt_id'=a.id::text)=1
		  AND EXISTS (SELECT 1 FROM outbox_events o WHERE o.aggregate_type='Command'
		              AND o.aggregate_id=c.id AND o.delivered_at IS NOT NULL)
		ORDER BY a.created_at,a.id LIMIT $1
	`, limit)
	if err != nil {
		return 0, err
	}
	type candidate struct {
		attemptID, commandID uuid.UUID
		commandType          string
		payload              []byte
	}
	var candidates []candidate
	for rows.Next() {
		var row candidate
		if err := rows.Scan(&row.attemptID, &row.commandID, &row.commandType, &row.payload); err != nil {
			rows.Close()
			return 0, err
		}
		candidates = append(candidates, row)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}
	closed := 0
	for _, row := range candidates {
		var payload map[string]any
		if err := json.Unmarshal(row.payload, &payload); err != nil {
			return closed, err
		}
		if payload["command_id"] != row.commandID.String() || payload["attempt_id"] != row.attemptID.String() {
			return closed, errors.New("RECONCILIATION_COMMAND_IDENTITY_DRIFT")
		}
		_, workflowID, _, _, err := workflowSpec(row.commandType, payload)
		if err != nil {
			return closed, err
		}
		response, err := temporalClient.DescribeWorkflowExecution(ctx, workflowID, "")
		if err != nil {
			// Missing/unreachable workflows are not proof of terminal failure.
			return closed, fmt.Errorf("describe bound workflow %s: %w", workflowID, err)
		}
		info := response.GetWorkflowExecutionInfo()
		if !unsuccessfulTerminal(info.GetStatus()) {
			continue
		}
		if info.GetExecution().GetWorkflowId() != workflowID || info.GetExecution().GetRunId() == "" || info.GetCloseTime() == nil {
			return closed, errors.New("RECONCILIATION_TERMINAL_WITNESS_INVALID")
		}
		evidence, _ := json.Marshal(map[string]any{
			"schema": "RealSaS.TerminalWorkflowFailure.v1", "command_id": row.commandID.String(),
			"workflow_id": workflowID, "workflow_run_id": info.GetExecution().GetRunId(),
			"workflow_status": info.GetStatus().String(), "workflow_closed_at": info.GetCloseTime().AsTime(),
		})
		changed := false
		err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
			changed = false
			var state string
			if err := tx.QueryRow(ctx, "SELECT final_state FROM attempts WHERE id=$1 FOR UPDATE", row.attemptID).Scan(&state); err != nil {
				return err
			}
			if state != "OPEN" {
				return nil
			}
			var count int
			if err := tx.QueryRow(ctx, "SELECT count(*) FROM commands WHERE payload->>'attempt_id'=$1", row.attemptID.String()).Scan(&count); err != nil {
				return err
			}
			if count != 1 {
				return errors.New("RECONCILIATION_COMMAND_SET_CHANGED")
			}
			if _, err := tx.Exec(ctx, "UPDATE attempts SET final_state='FAILED' WHERE id=$1", row.attemptID); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, "INSERT INTO attempt_events(attempt_id,event_type,payload) VALUES ($1,'ATTEMPT_WORKFLOW_FAILED',$2)", row.attemptID, evidence); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO audit_events(actor,action,attempt_id,payload)
				VALUES ('control-workflow-reconciler','ATTEMPT_WORKFLOW_FAILED',$1,$2)`, row.attemptID, evidence); err != nil {
				return err
			}
			changed = true
			return nil
		})
		if err != nil {
			return closed, err
		}
		if changed {
			closed++
		}
	}
	return closed, nil
}
