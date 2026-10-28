package campaign

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func componentGraph(t *testing.T, extraRoot bool, product bool) *stagegraph.Graph {
	t.Helper()
	stages := []stagegraph.Stage{
		{Ordinal: 1, ID: "SOURCE", DependsOn: []string{}},
		{Ordinal: 2, ID: "MEASUREMENT", DependsOn: []string{"SOURCE"}},
		{Ordinal: 3, ID: "TRAINING_WAIT", DependsOn: []string{"MEASUREMENT"}},
		{Ordinal: 4, ID: "PRODUCT", DependsOn: []string{"TRAINING_WAIT"}, Policy: stagegraph.Policy{ProductPassAuthority: product}},
	}
	if extraRoot {
		stages = append(stages, stagegraph.Stage{Ordinal: 5, ID: "OTHER_SOURCE", DependsOn: []string{}})
	}
	raw, err := json.Marshal(stagegraph.Snapshot{Schema: "RealSaS.StageGraph.v1", StageCount: len(stages), Stages: stages})
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestCampaignTargetPreservesDefaultAndBoundsExplicitComponent(t *testing.T) {
	g := componentGraph(t, false, true)
	for _, requested := range []string{"", "PRODUCT"} {
		target, scope, err := campaignTarget(g, requested)
		if err != nil || target != "PRODUCT" || scope != "FULL_DAG_FROM_SOURCE_ENTRY_TO_PRODUCT_CLOSURE" {
			t.Fatalf("default target=%s scope=%s err=%v", target, scope, err)
		}
	}
	target, scope, err := campaignTarget(g, "MEASUREMENT")
	if err != nil || target != "MEASUREMENT" || scope != "SOURCE_ENTRY_TO_RESEARCH_COMPONENT" {
		t.Fatalf("component target=%s scope=%s err=%v", target, scope, err)
	}
	closure, _ := g.AncestorsIncluding(target)
	if len(closure) != 2 {
		t.Fatalf("component must not enter downstream training wait: %v", closure)
	}
}

func TestCampaignTargetRejectsUnknownAndOmittedSourceRoots(t *testing.T) {
	_, _, err := campaignTarget(componentGraph(t, false, true), "UNKNOWN")
	if err == nil {
		t.Fatal("unknown target accepted")
	}
	_, _, err = campaignTarget(componentGraph(t, true, true), "MEASUREMENT")
	if err == nil || !strings.Contains(err.Error(), "COMPONENT_SOURCE_ENTRY_MISSING") {
		t.Fatalf("missing source root accepted: %v", err)
	}
	_, _, err = campaignTarget(componentGraph(t, true, true), "")
	if err == nil || !strings.Contains(err.Error(), "PRODUCT_TARGET_MUST_CLOSE_FULL_DAG") {
		t.Fatalf("default full target contract weakened: %v", err)
	}
}

func TestResearchGraphWithoutProductRequiresExplicitTarget(t *testing.T) {
	g := componentGraph(t, false, false)
	if _, _, err := campaignTarget(g, ""); err == nil {
		t.Fatal("implicit target accepted on component-only release")
	}
	if target, _, err := campaignTarget(g, "MEASUREMENT"); err != nil || target != "MEASUREMENT" {
		t.Fatalf("explicit component rejected: %v", err)
	}
}
