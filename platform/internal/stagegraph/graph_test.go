package stagegraph

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func canonicalPlan(t *testing.T) []byte {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestCanonicalGraphAndRuntimeInvalidation(t *testing.T) {
	g, err := ParseCanonicalPlan(canonicalPlan(t))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(g.Stages()); got != 46 {
		t.Fatalf("got %d stages", got)
	}
	affected, err := g.DescendantsIncluding("42_RUNTIME_PROJECTION_AND_CAA_BINDING")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"42_RUNTIME_PROJECTION_AND_CAA_BINDING",
		"43_RSS_MATERIALIZE_COMPACT",
		"44_NATIVE_PACKAGE_OPEN_PLAYBACK",
		"45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
		"46_PRODUCT_CLOSURE_SEAL",
	}
	if len(affected) != len(want) {
		t.Fatalf("affected=%v want=%v", affected, want)
	}
	for i := range want {
		if affected[i] != want[i] {
			t.Fatalf("affected=%v want=%v", affected, want)
		}
	}
}
