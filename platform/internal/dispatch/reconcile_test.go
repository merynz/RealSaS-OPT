package dispatch

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"go.temporal.io/api/common/v1"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/api/workflow/v1"
	"go.temporal.io/api/workflowservice/v1"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type describeWitness struct {
	response *workflowservice.DescribeWorkflowExecutionResponse
	err      error
	calls    int
}

func (d *describeWitness) DescribeWorkflowExecution(_ context.Context, id, run string) (*workflowservice.DescribeWorkflowExecutionResponse, error) {
	d.calls++
	if d.response != nil {
		d.response.WorkflowExecutionInfo.Execution.WorkflowId = id
	}
	return d.response, d.err
}

func TestUnsuccessfulTerminalExcludesUnknownRunningSuccessfulAndContinued(t *testing.T) {
	for _, status := range []enumspb.WorkflowExecutionStatus{
		enumspb.WORKFLOW_EXECUTION_STATUS_UNSPECIFIED, enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING,
		enumspb.WORKFLOW_EXECUTION_STATUS_COMPLETED, enumspb.WORKFLOW_EXECUTION_STATUS_CONTINUED_AS_NEW,
	} {
		if unsuccessfulTerminal(status) {
			t.Fatalf("cannot fail %s", status)
		}
	}
	for _, status := range []enumspb.WorkflowExecutionStatus{
		enumspb.WORKFLOW_EXECUTION_STATUS_FAILED, enumspb.WORKFLOW_EXECUTION_STATUS_CANCELED,
		enumspb.WORKFLOW_EXECUTION_STATUS_TERMINATED, enumspb.WORKFLOW_EXECUTION_STATUS_TIMED_OUT,
	} {
		if !unsuccessfulTerminal(status) {
			t.Fatalf("must reconcile %s", status)
		}
	}
}

func TestReconciliationRequiresTemporalTerminalWitnessAndPreservesEvidence(t *testing.T) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	subjectID, attemptID, commandID := uuid.New(), uuid.New(), uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,'Reconciliation test')", subjectID, subjectID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO attempts(id,subject_id,kind,spec_sha256,created_by) VALUES ($1,$2,'research',$3,'ci')", attemptID, subjectID, hex64()); err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(map[string]any{
		"command_id": commandID.String(), "attempt_id": attemptID.String(), "subject_id": subjectID.String(),
		"engine_release_id": uuid.NewString(), "subject_input_id": uuid.NewString(), "target_stage_id": "02_SOURCE_LICENSE_PROVENANCE",
	})
	if _, err := pool.Exec(ctx, "INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload) VALUES ($1,'COMPILE_SUBJECT',$2,$3,$4)", commandID, subjectID, commandID.String(), payload); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload) VALUES ('Command',$1,'CompileSubjectRequested','{}')", commandID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO attempt_events(attempt_id,event_type,payload) VALUES ($1,'FAILURE_LOCALIZED','{\"preserved\":true}')", attemptID); err != nil {
		t.Fatal(err)
	}
	witness := &describeWitness{response: &workflowservice.DescribeWorkflowExecutionResponse{WorkflowExecutionInfo: &workflow.WorkflowExecutionInfo{
		Execution: &common.WorkflowExecution{RunId: uuid.NewString()}, Status: enumspb.WORKFLOW_EXECUTION_STATUS_FAILED,
		CloseTime: timestamppb.New(time.Now()),
	}}}
	if count, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err != nil || count != 0 || witness.calls != 0 {
		t.Fatalf("undelivered command: count=%d calls=%d err=%v", count, witness.calls, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE outbox_events SET delivered_at=now() WHERE aggregate_id=$1", commandID); err != nil {
		t.Fatal(err)
	}
	for _, status := range []enumspb.WorkflowExecutionStatus{enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING, enumspb.WORKFLOW_EXECUTION_STATUS_COMPLETED} {
		witness.response.WorkflowExecutionInfo.Status = status
		if count, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err != nil || count != 0 {
			t.Fatalf("status %s count=%d err=%v", status, count, err)
		}
	}
	witness.err = errors.New("Temporal unavailable")
	if _, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err == nil {
		t.Fatal("unavailable Temporal must not infer failure")
	}
	witness.err = nil
	witness.response.WorkflowExecutionInfo.Status = enumspb.WORKFLOW_EXECUTION_STATUS_FAILED
	witness.response.WorkflowExecutionInfo.CloseTime = nil
	if _, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err == nil {
		t.Fatal("terminal status without close time is not proof")
	}
	witness.response.WorkflowExecutionInfo.CloseTime = timestamppb.New(time.Now())
	if count, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err != nil || count != 1 {
		t.Fatalf("terminal failure count=%d err=%v", count, err)
	}
	if count, err := ReconcileFailedAttempts(ctx, pool, witness, 32); err != nil || count != 0 {
		t.Fatalf("idempotence count=%d err=%v", count, err)
	}
	var state string
	if err := pool.QueryRow(ctx, "SELECT final_state FROM attempts WHERE id=$1", attemptID).Scan(&state); err != nil || state != "FAILED" {
		t.Fatalf("state=%s err=%v", state, err)
	}
	var events, failures, products int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM attempt_events WHERE attempt_id=$1 AND event_type='ATTEMPT_WORKFLOW_FAILED'", attemptID).Scan(&events); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM attempt_events WHERE attempt_id=$1 AND event_type='FAILURE_LOCALIZED' AND payload->>'preserved'='true'", attemptID).Scan(&failures); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM product_revisions WHERE subject_id=$1", subjectID).Scan(&products); err != nil {
		t.Fatal(err)
	}
	if events != 1 || failures != 1 || products != 0 {
		t.Fatalf("events=%d preserved=%d products=%d", events, failures, products)
	}
}

func hex64() string { return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
