package dispatch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	enumspb "go.temporal.io/api/enums/v1"
)

// Cancellation is a lifecycle operation, never scientific failure classification
// or qualification. Temporal terminal witnesses are required before closing OPEN
// Attempts, and every existing event/artifact remains untouched.
type CancelCampaignRequest struct {
	CampaignID              uuid.UUID   `json:"campaign_id"`
	ExpectedCampaignCodeSHA string      `json:"expected_campaign_code_sha"`
	ExpectedEngineReleaseID uuid.UUID   `json:"expected_engine_release_id"`
	ExpectedAttemptIDs      []uuid.UUID `json:"expected_attempt_ids"`
	RequestedBy             string      `json:"requested_by"`
	Reason                  string      `json:"reason"`
}

type WorkflowCanceler interface {
	WorkflowDescriber
	CancelWorkflow(context.Context, string, string) error
}

type CancellationBinding struct {
	AttemptID             uuid.UUID `json:"attempt_id"`
	CommandID             uuid.UUID `json:"command_id"`
	WorkflowID            string    `json:"workflow_id"`
	RunID                 string    `json:"workflow_run_id"`
	Status                string    `json:"workflow_status"`
	CancellationRequested bool      `json:"cancellation_requested"`
}

type CancelCampaignReceipt struct {
	Schema                 string                `json:"schema"`
	CampaignID             uuid.UUID             `json:"campaign_id"`
	RequestedBy            string                `json:"requested_by"`
	Reason                 string                `json:"reason"`
	Bindings               []CancellationBinding `json:"bindings"`
	ReconciledAttemptCount int                   `json:"reconciled_attempt_count"`
	TrainingAuthorized     bool                  `json:"training_authorized"`
}

func validateCancellationScope(req CancelCampaignRequest, actual []uuid.UUID) error {
	if req.CampaignID == uuid.Nil || req.ExpectedEngineReleaseID == uuid.Nil || len(req.ExpectedCampaignCodeSHA) != 40 ||
		strings.TrimSpace(req.RequestedBy) == "" || strings.TrimSpace(req.Reason) == "" || len(req.ExpectedAttemptIDs) == 0 {
		return errors.New("CAMPAIGN_CANCELLATION_EXPLICIT_SCOPE_REQUIRED")
	}
	seen := map[uuid.UUID]bool{}
	for _, id := range req.ExpectedAttemptIDs {
		if id == uuid.Nil || seen[id] {
			return errors.New("CAMPAIGN_CANCELLATION_DUPLICATE_OR_EMPTY_ATTEMPT")
		}
		seen[id] = true
	}
	before, after := slices.Clone(req.ExpectedAttemptIDs), slices.Clone(actual)
	less := func(a, b uuid.UUID) int { return strings.Compare(a.String(), b.String()) }
	slices.SortFunc(before, less)
	slices.SortFunc(after, less)
	if !slices.Equal(before, after) {
		return errors.New("CAMPAIGN_CANCELLATION_MEMBER_SCOPE_DRIFT")
	}
	return nil
}

func CancelResearchCampaign(ctx context.Context, pool *pgxpool.Pool, witness WorkflowCanceler, req CancelCampaignRequest) (CancelCampaignReceipt, error) {
	receipt := CancelCampaignReceipt{Schema: "RealSaS.ResearchCampaignCancellation.v1", CampaignID: req.CampaignID,
		RequestedBy: req.RequestedBy, Reason: req.Reason}
	if pool == nil || witness == nil {
		return receipt, errors.New("CAMPAIGN_CANCELLATION_OWNERS_REQUIRED")
	}
	var code, purpose string
	var releaseID uuid.UUID
	err := pool.QueryRow(ctx, `SELECT c.canonical_code_sha,c.candidate_engine_release_id,r.purpose
		FROM research_campaigns c JOIN engine_releases r ON r.id=c.candidate_engine_release_id WHERE c.id=$1`, req.CampaignID).Scan(&code, &releaseID, &purpose)
	if err != nil {
		return receipt, err
	}
	if code != req.ExpectedCampaignCodeSHA || releaseID != req.ExpectedEngineReleaseID || purpose != "RESEARCH" {
		return receipt, errors.New("CAMPAIGN_CANCELLATION_RELEASE_SCOPE_DRIFT")
	}
	rows, err := pool.Query(ctx, `SELECT a.id,a.subject_id,a.kind,a.engine_release_id FROM research_campaign_members m
		JOIN attempts a ON a.id=m.attempt_id WHERE m.campaign_id=$1 ORDER BY a.id`, req.CampaignID)
	if err != nil {
		return receipt, err
	}
	type member struct {
		id, subject, release uuid.UUID
		kind                 string
	}
	var members []member
	var ids []uuid.UUID
	for rows.Next() {
		var m member
		if err = rows.Scan(&m.id, &m.subject, &m.kind, &m.release); err != nil {
			rows.Close()
			return receipt, err
		}
		members = append(members, m)
		ids = append(ids, m.id)
	}
	rows.Close()
	if err = rows.Err(); err != nil {
		return receipt, err
	}
	if err = validateCancellationScope(req, ids); err != nil {
		return receipt, err
	}
	// Resolve and describe the complete exact scope before requesting any cancel.
	for _, m := range members {
		if m.kind != "research" || m.release != releaseID {
			return receipt, errors.New("CAMPAIGN_CANCELLATION_ATTEMPT_SCOPE_DRIFT")
		}
		commands, e := pool.Query(ctx, `SELECT c.id,c.command_type,c.payload,
			EXISTS(SELECT 1 FROM outbox_events o WHERE o.aggregate_type='Command' AND o.aggregate_id=c.id AND o.delivered_at IS NOT NULL)
			FROM commands c WHERE c.payload->>'attempt_id'=$1`, m.id.String())
		if e != nil {
			return receipt, e
		}
		count := 0
		var commandID uuid.UUID
		var kind string
		var raw []byte
		var delivered bool
		for commands.Next() {
			count++
			if e = commands.Scan(&commandID, &kind, &raw, &delivered); e != nil {
				commands.Close()
				return receipt, e
			}
		}
		commands.Close()
		if e = commands.Err(); e != nil {
			return receipt, e
		}
		if count != 1 || kind != "COMPILE_SUBJECT" || !delivered {
			return receipt, errors.New("CAMPAIGN_CANCELLATION_COMMAND_SCOPE_AMBIGUOUS_OR_UNDELIVERED")
		}
		var payload map[string]any
		if e = json.Unmarshal(raw, &payload); e != nil {
			return receipt, e
		}
		if payload["command_id"] != commandID.String() || payload["attempt_id"] != m.id.String() || payload["subject_id"] != m.subject.String() || payload["engine_release_id"] != releaseID.String() {
			return receipt, errors.New("CAMPAIGN_CANCELLATION_COMMAND_IDENTITY_DRIFT")
		}
		_, workflowID, _, _, e := workflowSpec(kind, payload)
		if e != nil {
			return receipt, e
		}
		description, e := witness.DescribeWorkflowExecution(ctx, workflowID, "")
		if e != nil {
			return receipt, e
		}
		info := description.GetWorkflowExecutionInfo()
		if info.GetExecution().GetWorkflowId() != workflowID || info.GetExecution().GetRunId() == "" || info.GetStatus() == enumspb.WORKFLOW_EXECUTION_STATUS_UNSPECIFIED {
			return receipt, errors.New("CAMPAIGN_CANCELLATION_WORKFLOW_IDENTITY_DRIFT")
		}
		if info.GetStatus() != enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING && info.GetCloseTime() == nil {
			return receipt, errors.New("CAMPAIGN_CANCELLATION_TERMINAL_WITNESS_INVALID")
		}
		receipt.Bindings = append(receipt.Bindings, CancellationBinding{AttemptID: m.id, CommandID: commandID, WorkflowID: workflowID, RunID: info.GetExecution().GetRunId(), Status: info.GetStatus().String()})
	}
	raw, err := json.Marshal(map[string]any{"request": req, "bindings": receipt.Bindings})
	if err != nil {
		return receipt, err
	}
	if _, err = pool.Exec(ctx, `INSERT INTO research_campaign_events(campaign_id,event_type,payload) VALUES ($1,'CAMPAIGN_CANCELLATION_REQUESTED',$2)`, req.CampaignID, raw); err != nil {
		return receipt, err
	}
	if _, err = pool.Exec(ctx, `INSERT INTO audit_events(actor,action,payload) VALUES ($1,'CAMPAIGN_CANCELLATION_REQUESTED',$2)`, req.RequestedBy, raw); err != nil {
		return receipt, err
	}
	for i, b := range receipt.Bindings {
		if b.Status == enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING.String() {
			if err = witness.CancelWorkflow(ctx, b.WorkflowID, b.RunID); err != nil {
				return receipt, err
			}
			receipt.Bindings[i].CancellationRequested = true
		}
	}
	for {
		terminal := true
		for i, b := range receipt.Bindings {
			d, e := witness.DescribeWorkflowExecution(ctx, b.WorkflowID, b.RunID)
			if e != nil {
				return receipt, e
			}
			info := d.GetWorkflowExecutionInfo()
			if info.GetExecution().GetWorkflowId() != b.WorkflowID || info.GetExecution().GetRunId() != b.RunID {
				return receipt, errors.New("CAMPAIGN_CANCELLATION_WORKFLOW_RUN_DRIFT")
			}
			receipt.Bindings[i].Status = info.GetStatus().String()
			if info.GetStatus() == enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING {
				terminal = false
			} else if info.GetCloseTime() == nil {
				return receipt, errors.New("CAMPAIGN_CANCELLATION_TERMINAL_WITNESS_INVALID")
			}
		}
		if terminal {
			break
		}
		select {
		case <-ctx.Done():
			return receipt, fmt.Errorf("CAMPAIGN_CANCELLATION_NOT_TERMINAL: %w", ctx.Err())
		case <-time.After(250 * time.Millisecond):
		}
	}
	receipt.ReconciledAttemptCount, err = ReconcileFailedAttemptIDs(ctx, pool, witness, ids)
	if err != nil {
		return receipt, err
	}
	raw, err = json.Marshal(receipt)
	if err != nil {
		return receipt, err
	}
	_, err = pool.Exec(ctx, `INSERT INTO research_campaign_events(campaign_id,event_type,payload) VALUES ($1,'CAMPAIGN_CANCELLATION_CONFIRMED',$2)`, req.CampaignID, raw)
	return receipt, err
}
