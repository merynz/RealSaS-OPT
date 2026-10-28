package campaign

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func campaignGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func campaignHex(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}

func campaignManifest(g *stagegraph.Graph, name string, purpose release.Purpose) release.Manifest {
	rows := make([]release.StageBinding, 0, g.StageCount())
	for _, s := range g.Stages() {
		rows = append(rows, release.StageBinding{
			Ordinal: s.Ordinal, StageID: s.ID,
			ImplementationSHA256: campaignHex("1"),
			PolicySHA256:         campaignHex("2"),
			SemanticParameters:   map[string]any{"campaign_test": true},
		})
	}
	return release.Manifest{Name: name, Purpose: purpose, Stages: rows}
}

func TestOrderedMembersRejectsOverlap(t *testing.T) {
	id := uuid.New()
	_, err := orderedMembers(StartRequest{CohortSubjectIDs: []uuid.UUID{id}, RegressionSubjectIDs: []uuid.UUID{id}})
	if err == nil {
		t.Fatal("expected duplicate subject rejection")
	}
}

func TestCampaignCreatesFullDAGAttemptsAndNeverMintsFit8Pass(t *testing.T) {
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

	g := campaignGraph(t)
	baseline, err := release.Seal(ctx, pool, g, campaignManifest(g, "campaign-base-"+uuid.New().String(), release.PurposeProduct), "ci")
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := release.Seal(ctx, pool, g, campaignManifest(g, "campaign-candidate-"+uuid.New().String(), release.PurposeResearch), "ci")
	if err != nil {
		t.Fatal(err)
	}
	canonicalCodeSHA := "0123456789abcdef0123456789abcdef01234567"
	if _, err := release.BindCanonicalMain(ctx, pool, candidate.ReleaseID, canonicalCodeSHA, "ci"); err != nil {
		t.Fatal(err)
	}

	subjects := []uuid.UUID{uuid.New(), uuid.New(), uuid.New()}
	for i, id := range subjects {
		if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", id, "campaign-"+id.String(), "Campaign subject"); err != nil {
			t.Fatal(i, err)
		}
	}
	status, err := Start(ctx, pool, g, canonicalCodeSHA, StartRequest{
		Name:                     "fit8-platform-test",
		BaselineEngineReleaseID:  baseline.ReleaseID,
		CandidateEngineReleaseID: candidate.ReleaseID,
		CohortSubjectIDs:         subjects[:2],
		RegressionSubjectIDs:     subjects[2:],
		CreatedBy:                "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if status.CanonicalCodeSHA != canonicalCodeSHA {
		t.Fatalf("campaign code sha=%s want=%s", status.CanonicalCodeSHA, canonicalCodeSHA)
	}
	if status.TargetStageID != "46_PRODUCT_CLOSURE_SEAL" {
		t.Fatalf("target=%s", status.TargetStageID)
	}
	if len(status.Members) != 3 {
		t.Fatalf("members=%d", len(status.Members))
	}
	if status.FullClosureComplete || status.Fit8PassClaimed {
		t.Fatalf("premature closure=%v pass=%v", status.FullClosureComplete, status.Fit8PassClaimed)
	}
	for _, m := range status.Members {
		if m.FinalState != "OPEN" {
			t.Fatalf("member %s state=%s", m.SubjectID, m.FinalState)
		}
		var memberCodeSHA string
		if err := pool.QueryRow(ctx, "SELECT canonical_code_sha FROM attempts WHERE id=$1", m.AttemptID).Scan(&memberCodeSHA); err != nil {
			t.Fatal(err)
		}
		if memberCodeSHA != canonicalCodeSHA {
			t.Fatalf("member %s code sha=%s want=%s", m.SubjectID, memberCodeSHA, canonicalCodeSHA)
		}
		if _, err := pool.Exec(ctx, "UPDATE attempts SET final_state='COMPLETED' WHERE id=$1", m.AttemptID); err != nil {
			t.Fatal(err)
		}
	}
	status, err = Get(ctx, pool, status.CampaignID)
	if err != nil {
		t.Fatal(err)
	}
	if !status.CohortComplete || !status.RegressionComplete || !status.FullClosureComplete || !status.GoalComplete {
		t.Fatalf("closure rollup=%+v", status)
	}
	if status.Fit8PassClaimed {
		t.Fatal("campaign must never mint Fit8 scientific PASS")
	}
	componentRequest := StartRequest{
		Name: "finite-source-measurement", BaselineEngineReleaseID: baseline.ReleaseID,
		CandidateEngineReleaseID: candidate.ReleaseID, TargetStageID: "08_NORMALIZATION_DOMAIN_QUALIFIED",
		CohortSubjectIDs: subjects[:2], RegressionSubjectIDs: subjects[2:], CreatedBy: "ci",
	}
	component, err := Start(ctx, pool, g, canonicalCodeSHA, componentRequest)
	if err != nil {
		t.Fatal(err)
	}
	if component.TargetStageID != componentRequest.TargetStageID || component.ExecutionScope != "SOURCE_ENTRY_TO_RESEARCH_COMPONENT" {
		t.Fatalf("component scope lost: %+v", component)
	}
	for _, member := range component.Members {
		if _, err := pool.Exec(ctx, "UPDATE attempts SET final_state='COMPLETED' WHERE id=$1", member.AttemptID); err != nil {
			t.Fatal(err)
		}
	}
	component, err = Get(ctx, pool, component.CampaignID)
	if err != nil {
		t.Fatal(err)
	}
	if !component.GoalComplete || component.FullClosureComplete || component.Fit8PassClaimed {
		t.Fatalf("finite target completion escalated to product closure: %+v", component)
	}
	componentRequest.TargetStageID = "46_PRODUCT_CLOSURE_SEAL"
	whole, err := Start(ctx, pool, g, canonicalCodeSHA, componentRequest)
	if err != nil {
		t.Fatal(err)
	}
	if whole.CampaignID == component.CampaignID {
		t.Fatal("different immutable target reused component campaign identity")
	}
}
