package agentsession

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestDurableEntryResumeExitRejectStaleContextAndScopeDrift(t *testing.T) {
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
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	reject := func(err error, want string) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("wanted %s: %v", want, err)
		}
	}
	sha := strings.Repeat("a", 64)
	code := strings.Repeat("b", 40)
	subject := uuid.New()
	_, err = pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,'Agent handoff fixture')", subject, "agent-"+subject.String())
	must(err)
	raw, _ := json.Marshal(stagegraph.Snapshot{StageCount: 2, Stages: []stagegraph.Stage{{Ordinal: 1, ID: "NEW_MODEL_HEAD"}, {Ordinal: 2, ID: "UNSEEN_EVALUATION", DependsOn: []string{"NEW_MODEL_HEAD"}}}})
	g, err := stagegraph.ParsePlan(raw, false)
	must(err)
	r, err := release.Seal(ctx, pool, g, release.Manifest{Name: "agent-" + uuid.NewString(), Purpose: release.PurposeResearch, Stages: []release.StageBinding{
		{Ordinal: 1, StageID: "NEW_MODEL_HEAD", ImplementationSHA256: sha, PolicySHA256: sha, SemanticParameters: map[string]any{}},
		{Ordinal: 2, StageID: "UNSEEN_EVALUATION", ImplementationSHA256: sha, PolicySHA256: sha, SemanticParameters: map[string]any{}},
	}}, "ci")
	must(err)
	var typeID uuid.UUID
	must(pool.QueryRow(ctx, `INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,'AgentSessionFixture','v1','test') ON CONFLICT(name,schema_version) DO UPDATE SET name=excluded.name RETURNING id`, uuid.New()).Scan(&typeID))
	artifact := uuid.New()
	artifactSHA := strings.ReplaceAll(artifact.String(), "-", "")
	artifactSHA += artifactSHA
	_, err = pool.Exec(ctx, `INSERT INTO artifacts(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,verified_at) VALUES ($1,$2,$3,$3,$4,0,'engineering_fixture',$3,$3,now())`, artifact, typeID, artifactSHA, "fixture/"+artifact.String())
	must(err)
	sealed, err := input.Seal(ctx, pool, subject, []input.Binding{{Role: "manifest:corpus", ArtifactID: artifact}}, "ci")
	must(err)
	newAttempt := func(parent *uuid.UUID) uuid.UUID {
		t.Helper()
		id := uuid.New()
		_, err := pool.Exec(ctx, `INSERT INTO attempts(id,subject_id,engine_release_id,parent_attempt_id,kind,spec_sha256,created_by,final_state) VALUES ($1,$2,$3,$4,'research',$5,'ci','OPEN')`, id, subject, r.ReleaseID, parent, sha)
		must(err)
		return id
	}
	attemptID := newAttempt(nil)
	req := OpenRequest{AttemptID: attemptID, SubjectInputID: sealed.SubjectInputID, TargetStageID: "NEW_MODEL_HEAD", CanonicalCodeSHA: code, CreatedBy: "agent-one"}
	ticket, err := Open(ctx, pool, code, req)
	must(err)
	_, err = Open(ctx, pool, code, req)
	reject(err, "ACTIVE_SESSION_MUST_RESUME")
	resume := req
	resume.ResumeSessionID = &ticket.Scope.SessionID
	resume.CreatedBy = "agent-two"
	resumed, err := Open(ctx, pool, code, resume)
	must(err)
	if !resumed.Resumed || resumed.ScopeSHA256 != ticket.ScopeSHA256 {
		t.Fatal("resume changed scope")
	}
	bad := resume
	bad.TargetStageID = "UNSEEN_EVALUATION"
	_, err = Open(ctx, pool, code, bad)
	reject(err, "RESUME_SCOPE_DRIFT")
	tx, err := pool.Begin(ctx)
	must(err)
	must(CheckCompile(ctx, tx, ticket.Scope.SessionID, attemptID, subject, r.ReleaseID, sealed.SubjectInputID, req.TargetStageID, code))
	must(tx.Rollback(ctx))
	tx, err = pool.Begin(ctx)
	must(err)
	err = CheckCompile(ctx, tx, ticket.Scope.SessionID, attemptID, subject, r.ReleaseID, sealed.SubjectInputID, "UNSEEN_EVALUATION", code)
	reject(err, "COMPILE_SCOPE_DRIFT")
	must(tx.Rollback(ctx))
	tx, err = pool.Begin(ctx)
	must(err)
	err = CheckCompile(ctx, tx, ticket.Scope.SessionID, attemptID, subject, r.ReleaseID, sealed.SubjectInputID, req.TargetStageID, strings.Repeat("c", 40))
	reject(err, "CANONICAL_CODE_MISMATCH")
	must(tx.Rollback(ctx))
	exit := CloseRequest{SessionID: ticket.Scope.SessionID, AttemptID: attemptID, ScopeSHA256: ticket.ScopeSHA256, NextAction: "Evaluate the new head on the separately sealed corpus", Summary: "Engineering fixture", CreatedBy: "agent-two"}
	badExit := exit
	badExit.ScopeSHA256 = sha
	_, err = Close(ctx, pool, badExit)
	reject(err, "EXIT_SCOPE_DRIFT")
	commandID := uuid.New()
	payload, _ := json.Marshal(map[string]string{"attempt_id": attemptID.String(), "agent_session_id": ticket.Scope.SessionID.String()})
	_, err = pool.Exec(ctx, `INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload) VALUES ($1,'COMPILE_SUBJECT',$2,$3,$4)`, commandID, subject, commandID.String(), payload)
	must(err)
	_, err = Close(ctx, pool, exit)
	reject(err, "COMMAND_STILL_RUNNING")
	_, err = pool.Exec(ctx, "UPDATE attempts SET final_state='COMPLETED' WHERE id=$1", attemptID)
	must(err)
	_, err = pool.Exec(ctx, "INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin) VALUES ($1,'stage:NEW_MODEL_HEAD',$2,'produced')", attemptID, artifact)
	must(err)
	handoff, err := Close(ctx, pool, exit)
	must(err)
	if handoff["artifact_bindings"].(map[string]string)["stage:NEW_MODEL_HEAD"] != artifact.String() {
		t.Fatal("handoff lost exact artifact")
	}
	again, err := Close(ctx, pool, exit)
	must(err)
	if again["handoff_sha256"] != handoff["handoff_sha256"] {
		t.Fatal("duplicate exit changed handoff")
	}
	badExit = exit
	badExit.NextAction = "Unrelated task"
	_, err = Close(ctx, pool, badExit)
	reject(err, "EXIT_IDEMPOTENCY_DRIFT")
	tx, err = pool.Begin(ctx)
	must(err)
	err = CheckCompile(ctx, tx, ticket.Scope.SessionID, attemptID, subject, r.ReleaseID, sealed.SubjectInputID, req.TargetStageID, code)
	reject(err, "CLOSED_OR_SUPERSEDED")
	must(tx.Rollback(ctx))
	child := newAttempt(&attemptID)
	next := req
	next.AttemptID = child
	next.TargetStageID = "UNSEEN_EVALUATION"
	_, err = Open(ctx, pool, code, next)
	reject(err, "STALE_HANDOFF")
	next.ExpectedHandoffSHA256 = handoff["handoff_sha256"].(string)
	unrelated := next
	unrelated.AttemptID = newAttempt(nil)
	_, err = Open(ctx, pool, code, unrelated)
	reject(err, "LINEAGE_DRIFT")
	childTicket, err := Open(ctx, pool, code, next)
	must(err)
	state, err := Context(ctx, pool, subject, code)
	must(err)
	if state["active_session"].(*Scope).SessionID != childTicket.Scope.SessionID || state["last_handoff"].(map[string]any)["handoff_sha256"] != handoff["handoff_sha256"] {
		t.Fatal("context lost durable continuation")
	}
}
