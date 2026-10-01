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
}

func repeat(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
