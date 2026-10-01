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
	if got := len(g.Stages()); got == 0 {
		t.Fatal("canonical graph must not be empty")
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

func TestPlanCardinalityIsDataNotCode(t *testing.T) {
	data := []byte(`{
	  "stage_count": 3,
	  "stages": [
	    {"ordinal":1,"id":"source","title":"source","group":"source","depends_on":[],"adapter":"x:y","policy":{"product_pass_authority":false}},
	    {"ordinal":2,"id":"model.infer","title":"infer","group":"model","depends_on":["source"],"adapter":"x:y","policy":{"product_pass_authority":false}},
	    {"ordinal":3,"id":"product.seal","title":"seal","group":"closure","depends_on":["model.infer"],"adapter":"x:y","policy":{"product_pass_authority":true}}
	  ]
	}`)
	g, err := ParseCanonicalPlan(data)
	if err != nil { t.Fatal(err) }
	if g.StageCount() != 3 { t.Fatalf("stage count=%d", g.StageCount()) }
	if id, ok := g.ProductPassStageID(); !ok || id != "product.seal" {
		t.Fatalf("product pass=%q ok=%v", id, ok)
	}
}
