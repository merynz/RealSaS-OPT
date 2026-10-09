package resolver

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestRGBInterventionPreservesMechanicalArtifactsAndRejectsColdCache(t *testing.T) {
	stages := []stagegraph.Stage{
		{Ordinal: 1, ID: "GEOMETRY", ManifestKeys: []string{"geometry"}},
		{Ordinal: 2, ID: "RIG", DependsOn: []string{"GEOMETRY"}},
		{Ordinal: 3, ID: "SKIN", DependsOn: []string{"GEOMETRY", "RIG"}},
		{Ordinal: 4, ID: "RGB", ManifestKeys: []string{"appearance"}},
		{Ordinal: 5, ID: "RENDER", DependsOn: []string{"SKIN", "RGB"}},
	}
	raw, _ := json.Marshal(stagegraph.Snapshot{StageCount: len(stages), Stages: stages})
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	roots := []domain.ArtifactInputIdentity{{Role: "manifest:geometry", ArtifactType: "Geometry", SemanticSHA256: hex64("a")}, {Role: "manifest:appearance", ArtifactType: "RGB", SemanticSHA256: hex64("b")}}
	versions := resolverVersions(g, hex64("1"))
	cache := newMemoryCatalog()
	base, err := Resolve(context.Background(), g, cache, "RENDER", roots, versions)
	if err != nil {
		t.Fatal(err)
	}
	cache.admit(base)
	baseline := map[string]uuid.UUID{}
	for _, row := range base.Stages {
		baseline[row.StageID] = cache.rows[row.ExpectedSemanticSHA256]
	}
	v := versions["RGB"]
	v.ImplementationSHA256 = hex64("2")
	versions["RGB"] = v
	impact, err := release.Compare(g, resolverVersions(g, hex64("1")), versions)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Resolve(context.Background(), g, cache, "RENDER", roots, versions)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := VerifyIntervention(plan, impact.UnchangedStageIDs, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.PreservedArtifacts) != 3 || !reflect.DeepEqual(receipt.ExecuteStageIDs, []string{"RGB", "RENDER"}) {
		t.Fatalf("receipt=%+v", receipt)
	}
	// Missing unrelated cache entries may normally be recomputed. A causal A/B must stop instead.
	delete(cache.rows, base.Stages[0].ExpectedSemanticSHA256)
	cold, _ := Resolve(context.Background(), g, cache, "RENDER", roots, versions)
	if _, err := VerifyIntervention(cold, impact.UnchangedStageIDs, baseline); err == nil {
		t.Fatal("cold mechanical cache silently became inference")
	}
	cache.rows[base.Stages[0].ExpectedSemanticSHA256] = uuid.New()
	replaced, _ := Resolve(context.Background(), g, cache, "RENDER", roots, versions)
	if _, err := VerifyIntervention(replaced, impact.UnchangedStageIDs, baseline); err == nil {
		t.Fatal("different baseline artifact accepted")
	}
	cache.rows[base.Stages[0].ExpectedSemanticSHA256] = baseline["GEOMETRY"]
	roots[0].SemanticSHA256 = hex64("c")
	drift, _ := Resolve(context.Background(), g, cache, "RENDER", roots, versions)
	if _, err := VerifyIntervention(drift, impact.UnchangedStageIDs, baseline); err == nil {
		t.Fatal("undeclared mechanical input drift accepted")
	}
	partial := Plan{TargetStageID: "RGB", Stages: plan.Stages[3:4]}
	outside, err := VerifyIntervention(partial, impact.UnchangedStageIDs, nil)
	if err != nil || len(outside.OutsideTargetStageIDs) != 3 || len(outside.PreservedArtifacts) != 0 {
		t.Fatalf("outside target falsely counted as exact reuse: %+v %v", outside, err)
	}
}
