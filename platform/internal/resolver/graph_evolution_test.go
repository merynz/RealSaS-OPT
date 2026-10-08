package resolver

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestGraphInsertionReusesIndependentArtifactsWithExactIDs(t *testing.T) {
	parse := func(stages []stagegraph.Stage) *stagegraph.Graph {
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
	old := parse([]stagegraph.Stage{{ID: "MODEL"}, {ID: "CAA"}, {ID: "COMPILER", DependsOn: []string{"MODEL"}}, {ID: "END", DependsOn: []string{"COMPILER", "CAA"}}})
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "Image", SemanticSHA256: hex64("a")}}
	cache := newMemoryCatalog()
	base, err := Resolve(context.Background(), old, cache, "END", root, resolverVersions(old, hex64("1")))
	if err != nil {
		t.Fatal(err)
	}
	cache.admit(base)
	next := parse([]stagegraph.Stage{{ID: "MODEL"}, {ID: "NEW_HEAD", DependsOn: []string{"MODEL"}}, {ID: "CAA"}, {ID: "COMPILER", DependsOn: []string{"NEW_HEAD"}}, {ID: "END", DependsOn: []string{"COMPILER", "CAA"}}})
	plan, err := Resolve(context.Background(), next, cache, "END", root, resolverVersions(next, hex64("1")))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.ExecuteStageIDs(), []string{"NEW_HEAD", "COMPILER", "END"}) {
		t.Fatalf("execute=%v", plan.ExecuteStageIDs())
	}
	for _, row := range plan.Stages {
		if row.Action == Reuse && (row.ReusableArtifactID == nil || *row.ReusableArtifactID != cache.rows[row.ExpectedSemanticSHA256]) {
			t.Fatal("independent artifact identity changed")
		}
	}
}
