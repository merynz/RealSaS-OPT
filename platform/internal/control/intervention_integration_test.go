package control

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/resolver"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestControlledInterventionPersistsExactReuseAndStopsScopeDrift(t *testing.T) {
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
	subjectID, parentID := uuid.New(), uuid.New()
	_, err = pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,'Intervention')", subjectID, "intervention-"+subjectID.String())
	must(err)
	snapshot := stagegraph.Snapshot{StageCount: 3, Stages: []stagegraph.Stage{
		{Ordinal: 1, ID: "MECHANICS"}, {Ordinal: 2, ID: "RGB"},
		{Ordinal: 3, ID: "RENDER", DependsOn: []string{"MECHANICS", "RGB"}},
	}}
	raw, _ := json.Marshal(snapshot)
	g, err := stagegraph.ParsePlan(raw, false)
	must(err)
	makeRelease := func(purpose release.Purpose, rgb string) release.Sealed {
		rows := []release.StageBinding{}
		for _, s := range g.Stages() {
			impl := repeatHex("1")
			if s.ID == "RGB" {
				impl = rgb
			}
			rows = append(rows, release.StageBinding{Ordinal: s.Ordinal, StageID: s.ID, ImplementationSHA256: impl, PolicySHA256: repeatHex("2"), SemanticParameters: map[string]any{}})
		}
		r, err := release.Seal(ctx, pool, g, release.Manifest{Name: "intervention-" + uuid.NewString(), Purpose: purpose, Stages: rows}, "ci")
		must(err)
		return r
	}
	base := makeRelease(release.PurposeProduct, repeatHex("1"))
	candidate := makeRelease(release.PurposeResearch, repeatHex("3"))
	_, err = pool.Exec(ctx, `INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')`, parentID, subjectID, base.ReleaseID, repeatHex("4"))
	must(err)
	insertArtifact := func(kind, semanticSHA string) uuid.UUID {
		var typeID uuid.UUID
		err := pool.QueryRow(ctx, `INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','test')
			ON CONFLICT (name,schema_version) DO UPDATE SET name=excluded.name RETURNING id`, uuid.New(), kind).Scan(&typeID)
		must(err)
		id := uuid.New()
		content := fmt.Sprintf("%x", sha256.Sum256([]byte(id.String())))
		_, err = pool.Exec(ctx, `INSERT INTO artifacts(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,verified_at)
			VALUES ($1,$2,$3,$4,$5,0,'test',$6,$7,now())`, id, typeID, semanticSHA, content, "test/"+content, repeatHex("1"), repeatHex("2"))
		must(err)
		return id
	}
	rootSHA := fmt.Sprintf("%x", sha256.Sum256([]byte(subjectID.String())))
	sourceID := insertArtifact("InterventionSource", rootSHA)
	sealedInput, err := input.Seal(ctx, pool, subjectID, []input.Binding{{Role: "source", ArtifactID: sourceID}}, "ci")
	must(err)
	versions, err := release.LoadVersions(ctx, pool, base.ReleaseID, true)
	must(err)
	basePlan, err := resolver.Resolve(ctx, g, registry.QualifiedCatalog{Pool: pool}, "RENDER", []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "InterventionSource", SemanticSHA256: rootSHA}}, versions)
	must(err)
	mechanicalID := insertArtifact(resolver.StageResultArtifactType, basePlan.Stages[0].ExpectedSemanticSHA256)
	_, err = pool.Exec(ctx, `INSERT INTO qualifications(id,artifact_id,qualification_type,result) VALUES ($1,$2,'REUSE_ELIGIBLE','PASS')`, uuid.New(), mechanicalID)
	must(err)
	_, err = pool.Exec(ctx, `INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin) VALUES ($1,'stage:MECHANICS',$2,'produced')`, parentID, mechanicalID)
	must(err)
	req := attempt.ResearchRequest{SubjectID: subjectID, BaselineEngineReleaseID: base.ReleaseID, CandidateEngineReleaseID: candidate.ReleaseID, ParentAttemptID: &parentID, CreatedBy: "ci", Intervention: &release.Intervention{DirectChangedStageIDs: []string{"RGB"}, PreservedStageIDs: []string{"MECHANICS"}}}
	bad := req
	bad.Intervention = &release.Intervention{DirectChangedStageIDs: []string{"RGB", "MECHANICS"}}
	if _, err := attempt.StartResearch(ctx, pool, g, bad); err == nil || !strings.Contains(err.Error(), "DIRECT_CHANGE_DRIFT") {
		t.Fatalf("scope drift: %v", err)
	}
	bad = req
	bad.ParentAttemptID = nil
	if _, err := attempt.StartResearch(ctx, pool, g, bad); err == nil || !strings.Contains(err.Error(), "PARENT_ATTEMPT_REQUIRED") {
		t.Fatalf("missing parent: %v", err)
	}
	bad = req
	bad.BaselineEngineReleaseID = candidate.ReleaseID
	bad.Intervention = &release.Intervention{}
	if _, err := attempt.StartResearch(ctx, pool, g, bad); err == nil || !strings.Contains(err.Error(), "PARENT_RELEASE_MISMATCH") {
		t.Fatalf("wrong baseline release: %v", err)
	}
	started, err := attempt.StartResearch(ctx, pool, g, req)
	must(err)
	a := Activities{Pool: pool, Graph: g}
	in := orchestration.CompileWorkflowInput{ExecutionMode: "RESEARCH", CommandID: uuid.NewString(), AttemptID: started.AttemptID.String(), SubjectID: subjectID.String(), EngineReleaseID: candidate.ReleaseID.String(), SubjectInputID: sealedInput.SubjectInputID.String(), TargetStageID: "RENDER"}
	plan, err := a.ResolveCompilePlan(ctx, in)
	must(err)
	if plan.Stages[0].Action != "REUSE" || plan.Stages[0].ReusableArtifactID == nil || *plan.Stages[0].ReusableArtifactID != mechanicalID.String() {
		t.Fatalf("mechanics changed: %+v", plan)
	}
	_, err = a.ResolveCompilePlan(ctx, in) // Activity retry does not multiply identical receipts.
	must(err)
	var count int
	var payload []byte
	must(pool.QueryRow(ctx, `SELECT count(*) FROM attempt_events WHERE attempt_id=$1 AND event_type='INTERVENTION_REUSE_VERIFIED'`, started.AttemptID).Scan(&count))
	if count != 1 {
		t.Fatalf("receipts=%d", count)
	}
	must(pool.QueryRow(ctx, `SELECT payload FROM attempt_events WHERE attempt_id=$1 AND event_type='INTERVENTION_REUSE_VERIFIED'`, started.AttemptID).Scan(&payload))
	if !strings.Contains(string(payload), mechanicalID.String()) {
		t.Fatal("receipt omitted exact baseline identity")
	}
	// Even a forged activity scope cannot execute or rebind the frozen stage.
	_, err = a.PrepareStageExecution(ctx, orchestration.PrepareStageExecutionRequest{AttemptID: in.AttemptID, SubjectID: in.SubjectID, EngineReleaseID: in.EngineReleaseID, StageID: "MECHANICS", AllowedExecuteStageIDs: []string{"MECHANICS"}})
	if err == nil || !strings.Contains(err.Error(), "FROZEN_STAGE_EXECUTION") {
		t.Fatalf("frozen execution: %v", err)
	}
	_, err = a.BindReusedStage(ctx, orchestration.BindReusedStageRequest{AttemptID: in.AttemptID, StageID: "MECHANICS", ArtifactID: sourceID.String()})
	if err == nil || !strings.Contains(err.Error(), "EXACT_REUSE_REQUIRED") {
		t.Fatalf("baseline rebind: %v", err)
	}
	// Qualification loss stops the experiment before an execution row can be created.
	_, err = pool.Exec(ctx, `DELETE FROM qualifications WHERE artifact_id=$1`, mechanicalID)
	must(err)
	_, err = a.ResolveCompilePlan(ctx, in)
	if err == nil || !strings.Contains(err.Error(), "EXACT_REUSE_REQUIRED") {
		t.Fatalf("cache miss became inference: %v", err)
	}
	must(pool.QueryRow(ctx, `SELECT count(*) FROM executions WHERE attempt_id=$1`, started.AttemptID).Scan(&count))
	if count != 0 {
		t.Fatal("engine execution began despite intervention failure")
	}
	frozen, err := attempt.LoadIntervention(ctx, pool, started.AttemptID)
	must(err)
	if frozen.BaselineArtifactIDs["MECHANICS"] != mechanicalID {
		t.Fatal("durable baseline pin missing")
	}
}
