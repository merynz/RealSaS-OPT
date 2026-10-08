package release

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func smallGraph(t *testing.T, extra bool) *stagegraph.Graph {
	t.Helper()
	stages := []stagegraph.Stage{{ID: "A", Adapter: "a.py"}}
	parent := "A"
	if extra {
		stages = append(stages, stagegraph.Stage{ID: "X", Adapter: "x.py", DependsOn: []string{"A"}})
		parent = "X"
	}
	stages = append(stages, stagegraph.Stage{ID: "B", Adapter: "b.py"},
		stagegraph.Stage{ID: "C", Adapter: "c.py", DependsOn: []string{parent}},
		stagegraph.Stage{ID: "D", Adapter: "d.py", DependsOn: []string{"B"}},
		stagegraph.Stage{ID: "END", Adapter: "end.py", DependsOn: []string{"C", "D"}})
	for i := range stages {
		stages[i].Ordinal = i + 1
	}
	raw, _ := json.Marshal(stagegraph.Snapshot{StageCount: len(stages), Stages: stages})
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func sameVersions(g *stagegraph.Graph) map[string]StageVersion {
	rows := map[string]StageVersion{}
	for _, stage := range g.Stages() {
		rows[stage.ID] = StageVersion{ImplementationSHA256: "impl", PolicySHA256: "policy", ParametersSHA256: "params"}
	}
	return rows
}

func TestInsertedModuleOnlyInvalidatesItsConsumers(t *testing.T) {
	old, next := smallGraph(t, false), smallGraph(t, true)
	impact, err := CompareGraphs(old, next, sameVersions(old), sameVersions(next))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(impact.InvalidatedStageIDs, []string{"X", "C", "END"}) {
		t.Fatalf("impact=%+v", impact)
	}
	if !reflect.DeepEqual(impact.UnchangedStageIDs, []string{"A", "B", "D"}) {
		t.Fatalf("independent stages lost: %+v", impact)
	}
	if len(impact.DirectChanges) != 2 || !impact.DirectChanges[0].Added || !impact.DirectChanges[1].GraphChanged {
		t.Fatalf("direct=%+v", impact.DirectChanges)
	}
}

func TestRemovedModuleUsesOldAndNewGraphClosures(t *testing.T) {
	old, next := smallGraph(t, true), smallGraph(t, false)
	impact, err := CompareGraphs(old, next, sameVersions(old), sameVersions(next))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(impact.RemovedStageIDs, []string{"X"}) || !reflect.DeepEqual(impact.InvalidatedStageIDs, []string{"C", "END"}) {
		t.Fatalf("impact=%+v", impact)
	}
}

func TestNodeDigestMatchesPythonAndIgnoresNavigation(t *testing.T) {
	stage := stagegraph.Stage{Ordinal: 1, ID: "A", Adapter: "x.py", ManifestKeys: []string{"motion"}, Policy: stagegraph.Policy{Cacheable: true, FailClosed: true}}
	got, err := stage.NodeSHA256()
	if err != nil {
		t.Fatal(err)
	}
	if got != "49499b357bd5bb298ddd6a254d1e7f25a8586336039aa9e01fbd163eeda61703" {
		t.Fatalf("cross-language digest=%s", got)
	}
	stage.Ordinal = 99
	stage.Title = "navigation changed"
	stable, _ := stage.NodeSHA256()
	if stable != got {
		t.Fatal("navigation invalidates component identity")
	}
}
