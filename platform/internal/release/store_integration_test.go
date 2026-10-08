package release

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func loadGraph(t *testing.T) *stagegraph.Graph {
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

func TestEngineReleaseIsImmutableAndReusable(t *testing.T) {
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
	g := loadGraph(t)

	rows := make([]StageBinding, 0, 46)
	for _, s := range g.Stages() {
		rows = append(rows, StageBinding{
			Ordinal:              s.Ordinal,
			StageID:              s.ID,
			ImplementationSHA256: repeat("1"),
			PolicySHA256:         repeat("2"),
			SemanticParameters:   map[string]any{"contract_epoch": 1},
		})
	}
	m := Manifest{Name: "release-store-ci", Purpose: PurposeProduct, Stages: rows}
	first, err := Seal(ctx, pool, g, m, "ci")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Seal(ctx, pool, g, m, "ci")
	if err != nil {
		t.Fatal(err)
	}
	if !second.Reused || first.ReleaseID != second.ReleaseID || first.ReleaseSHA256 != second.ReleaseSHA256 {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	versions, err := LoadVersions(ctx, pool, first.ReleaseID, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(versions) != 46 {
		t.Fatalf("versions=%d", len(versions))
	}
	sealedGraph, _, err := LoadGraph(ctx, pool, first.ReleaseID)
	if err != nil || sealedGraph.StageCount() != g.StageCount() {
		t.Fatalf("graph=%v error=%v", sealedGraph, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE engine_release_graphs SET pipeline_plan_sha256=$2 WHERE release_id=$1", first.ReleaseID, repeat("3")); err == nil {
		t.Fatal("sealed DAG mutation was accepted")
	}
	changed := g.Snapshot()
	changed.Stages = append(changed.Stages, stagegraph.Stage{Ordinal: 47, ID: "47_RESEARCH_NEW_MODULE", Adapter: "new.py", DependsOn: []string{g.Stages()[0].ID}})
	changed.StageCount++
	m.Name = "release-store-expanded-research-ci"
	m.Purpose = PurposeResearch
	m.DAG = &changed
	m.Stages = append(m.Stages, StageBinding{Ordinal: 47, StageID: "47_RESEARCH_NEW_MODULE", ImplementationSHA256: repeat("4"), PolicySHA256: repeat("5")})
	expanded, err := Seal(ctx, pool, g, m, "ci")
	if err != nil {
		t.Fatal(err)
	}
	newVersions, err := LoadVersions(ctx, pool, expanded.ReleaseID, false)
	if err != nil || len(newVersions) != 47 {
		t.Fatalf("new versions=%d error=%v", len(newVersions), err)
	}
	oldAgain, _, err := LoadGraph(ctx, pool, first.ReleaseID)
	if err != nil || oldAgain.StageCount() != 46 {
		t.Fatal("old release graph changed")
	}
}

func repeat(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
