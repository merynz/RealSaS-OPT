package dispatch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	commonpb "go.temporal.io/api/common/v1"
	enumspb "go.temporal.io/api/enums/v1"
	workflowpb "go.temporal.io/api/workflow/v1"
	"go.temporal.io/api/workflowservice/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type campaignCancelWitness struct {
	workflows map[string]*workflowservice.DescribeWorkflowExecutionResponse
	canceled  int
}

func (w *campaignCancelWitness) DescribeWorkflowExecution(_ context.Context, id, run string) (*workflowservice.DescribeWorkflowExecutionResponse, error) {
	return w.workflows[id], nil
}
func (w *campaignCancelWitness) CancelWorkflow(_ context.Context, id, run string) error {
	info := w.workflows[id].WorkflowExecutionInfo
	if info.Execution.RunId != run {
		panic("cancel must bind exact run")
	}
	w.canceled++
	info.Status = enumspb.WORKFLOW_EXECUTION_STATUS_CANCELED
	info.CloseTime = timestamppb.Now()
	return nil
}

func TestCancellationRejectsPartialOrDuplicateMemberScope(t *testing.T) {
	a, b := uuid.New(), uuid.New()
	r := CancelCampaignRequest{CampaignID: uuid.New(), ExpectedEngineReleaseID: uuid.New(), ExpectedCampaignCodeSHA: strings.Repeat("a", 40), ExpectedAttemptIDs: []uuid.UUID{a}, RequestedBy: "ci", Reason: "repair"}
	if validateCancellationScope(r, []uuid.UUID{a, b}) == nil {
		t.Fatal("cannot silently cancel a subset")
	}
	r.ExpectedAttemptIDs = []uuid.UUID{a, a}
	if validateCancellationScope(r, []uuid.UUID{a, a}) == nil {
		t.Fatal("duplicate scope")
	}
}

func TestCampaignCancellationBindsTemporalRunsPreservesCompletedMemberAndEvidence(t *testing.T) {
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
	releaseID, campaignID := uuid.New(), uuid.New()
	code := strings.Repeat("a", 40)
	releaseDigest := sha256.Sum256([]byte(releaseID.String()))
	if _, err = pool.Exec(ctx, `INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at) VALUES ($1,$2,$3,'RESEARCH','ci',now())`, releaseID, "cancel-"+releaseID.String(), hex.EncodeToString(releaseDigest[:])); err != nil {
		t.Fatal(err)
	}
	if _, err = release.BindCanonicalMain(ctx, pool, releaseID, code, "ci"); err != nil {
		t.Fatal(err)
	}
	unique := strings.Repeat("a", 32) + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err = pool.Exec(ctx, `INSERT INTO research_campaigns(id,name,baseline_engine_release_id,candidate_engine_release_id,target_stage_id,canonical_code_sha,spec_sha256,state,created_by) VALUES ($1,'cancel test',$2,$2,'01_SOURCE_BYTES_SEALED',$3,$4,'RUNNING','ci')`, campaignID, releaseID, code, unique); err != nil {
		t.Fatal(err)
	}
	witness := &campaignCancelWitness{workflows: map[string]*workflowservice.DescribeWorkflowExecutionResponse{}}
	req := CancelCampaignRequest{CampaignID: campaignID, ExpectedEngineReleaseID: releaseID, ExpectedCampaignCodeSHA: code, RequestedBy: "ci", Reason: "isolate unsafe native state"}
	for i, state := range []string{"OPEN", "COMPLETED"} {
		subjectID, attemptID, commandID := uuid.New(), uuid.New(), uuid.New()
		req.ExpectedAttemptIDs = append(req.ExpectedAttemptIDs, attemptID)
		if _, err = pool.Exec(ctx, `INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,'cancel fixture')`, subjectID, subjectID.String()); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state,canonical_code_sha) VALUES ($1,$2,$3,'research',$4,'ci',$5,$6)`, attemptID, subjectID, releaseID, hex64(), state, code); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO research_campaign_members(campaign_id,subject_id,attempt_id,member_role,ordinal) VALUES ($1,$2,$3,'COHORT',$4)`, campaignID, subjectID, attemptID, i); err != nil {
			t.Fatal(err)
		}
		payload, _ := json.Marshal(map[string]any{"command_id": commandID.String(), "attempt_id": attemptID.String(), "subject_id": subjectID.String(), "engine_release_id": releaseID.String(), "subject_input_id": uuid.NewString(), "target_stage_id": "01_SOURCE_BYTES_SEALED"})
		if _, err = pool.Exec(ctx, `INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload) VALUES ($1,'COMPILE_SUBJECT',$2,$3,$4)`, commandID, subjectID, commandID.String(), payload); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload,delivered_at) VALUES ('Command',$1,'CompileSubjectRequested','{}',now())`, commandID); err != nil {
			t.Fatal(err)
		}
		if _, err = pool.Exec(ctx, `INSERT INTO attempt_events(attempt_id,event_type,payload) VALUES ($1,'PRIOR_MEASUREMENT_EVIDENCE','{"preserve":true}')`, attemptID); err != nil {
			t.Fatal(err)
		}
		workflowID := "realsas:compile:" + commandID.String()
		info := &workflowpb.WorkflowExecutionInfo{Execution: &commonpb.WorkflowExecution{WorkflowId: workflowID, RunId: uuid.NewString()}, Status: enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING}
		if state == "COMPLETED" {
			info.Status = enumspb.WORKFLOW_EXECUTION_STATUS_COMPLETED
			info.CloseTime = timestamppb.New(time.Now())
		}
		witness.workflows[workflowID] = &workflowservice.DescribeWorkflowExecutionResponse{WorkflowExecutionInfo: info}
	}
	bad := req
	bad.ExpectedAttemptIDs = req.ExpectedAttemptIDs[:1]
	if _, err = CancelResearchCampaign(ctx, pool, witness, bad); err == nil || witness.canceled != 0 {
		t.Fatal("scope drift must fail before cancellation")
	}
	r, err := CancelResearchCampaign(ctx, pool, witness, req)
	if err != nil {
		t.Fatal(err)
	}
	if witness.canceled != 1 || r.ReconciledAttemptCount != 1 {
		t.Fatalf("cancel=%d reconciled=%d", witness.canceled, r.ReconciledAttemptCount)
	}
	for i, want := range []string{"FAILED", "COMPLETED"} {
		var state string
		var evidence int
		if err = pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, req.ExpectedAttemptIDs[i]).Scan(&state); err != nil {
			t.Fatal(err)
		}
		if err = pool.QueryRow(ctx, `SELECT count(*) FROM attempt_events WHERE attempt_id=$1 AND event_type='PRIOR_MEASUREMENT_EVIDENCE'`, req.ExpectedAttemptIDs[i]).Scan(&evidence); err != nil {
			t.Fatal(err)
		}
		if state != want || evidence != 1 {
			t.Fatalf("state=%s evidence=%d", state, evidence)
		}
	}
	if _, err = CancelResearchCampaign(ctx, pool, witness, req); err != nil {
		t.Fatal(err)
	}
	if witness.canceled != 1 {
		t.Fatal("completed cancellation must not resubmit execution")
	}
}
