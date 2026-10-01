package attempt

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func testGraph(t *testing.T) *stagegraph.Graph {
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

func manifest(g *stagegraph.Graph, name string, purpose release.Purpose, stage35Impl string) release.Manifest {
	rows := make([]release.StageBinding, 0, 46)
	for _, s := range g.Stages() {
		impl := hexRepeat("1")
		if s.ID == "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
			impl = stage35Impl
		}
		rows = append(rows, release.StageBinding{
			Ordinal:              s.Ordinal,
			StageID:              s.ID,
			ImplementationSHA256: impl,
			PolicySHA256:         hexRepeat("2"),
			SemanticParameters:   map[string]any{"contract_epoch": 1},
		})
	}
	return release.Manifest{Name: name, Purpose: purpose, Stages: rows}
}

func TestResearchAttemptCapturesExactStage35Impact(t *testing.T) {
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

	g := testGraph(t)
	subjectID := uuid.New()
	if _, err := pool.Exec(ctx,
		"INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)",
		subjectID, "research-"+subjectID.String(), "Research Impact",
	); err != nil {
		t.Fatal(err)
	}
	baseline, err := release.Seal(ctx, pool, g, manifest(g, "baseline-"+subjectID.String(), release.PurposeProduct, hexRepeat("1")), "ci")
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := release.Seal(ctx, pool, g, manifest(g, "candidate-"+subjectID.String(), release.PurposeResearch, hexRepeat("3")), "ci")
	if err != nil {
		t.Fatal(err)
	}
	started, err := StartResearch(ctx, pool, g, ResearchRequest{
		SubjectID:                subjectID,
		BaselineEngineReleaseID:  baseline.ReleaseID,
		CandidateEngineReleaseID: candidate.ReleaseID,
		CreatedBy:                "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(started.Impact.DirectChanges) != 1 || started.Impact.DirectChanges[0].StageID != "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
		t.Fatalf("direct changes=%v", started.Impact.DirectChanges)
	}
	wantInvalidated, err := g.DescendantsIncluding("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED")
	if err != nil {
		t.Fatal(err)
	}
	if !sameStrings(started.Impact.InvalidatedStageIDs, wantInvalidated) {
		t.Fatalf("invalidated=%v want=%v", started.Impact.InvalidatedStageIDs, wantInvalidated)
	}
	for _, id := range started.Impact.UnchangedStageIDs {
		if id == "10_IRIS_FIT" {
			goto found
		}
	}
	t.Fatal("IRIS fit should remain unchanged")
found:

	var releaseID uuid.UUID
	if err := pool.QueryRow(ctx,
		"SELECT engine_release_id FROM attempts WHERE id=$1",
		started.AttemptID,
	).Scan(&releaseID); err != nil {
		t.Fatal(err)
	}
	if releaseID != candidate.ReleaseID {
		t.Fatalf("attempt release=%s candidate=%s", releaseID, candidate.ReleaseID)
	}

	var payload []byte
	if err := pool.QueryRow(ctx, `
		SELECT payload
		FROM attempt_events
		WHERE attempt_id=$1 AND event_type='CODE_CHANGE_IMPACT_RESOLVED'
	`, started.AttemptID).Scan(&payload); err != nil {
		t.Fatal(err)
	}
	var event map[string]any
	if err := json.Unmarshal(payload, &event); err != nil {
		t.Fatal(err)
	}
	direct, ok := event["direct_changed_stage_ids"].([]any)
	if !ok || len(direct) != 1 || direct[0] != "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
		t.Fatalf("event direct changes=%v", event["direct_changed_stage_ids"])
	}
}

func hexRepeat(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
