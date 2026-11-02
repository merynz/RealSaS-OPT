package attempt

import (
	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/resolver"
	"testing"
)

func TestPartialParentPreservesActualArtifactsAndMayComputeMissingPrefix(t *testing.T) {
	id := uuid.New()
	frozen := FrozenIntervention{FrozenStageIDs: []string{"A", "B"}, BaselineArtifactIDs: map[string]uuid.UUID{"A": id}}
	frozen.PinAvailableStages()
	if len(frozen.FrozenStageIDs) != 1 || frozen.FrozenStageIDs[0] != "A" {
		t.Fatalf("unproduced B must not enter immutable frozen set: %+v", frozen)
	}
	if frozen.CheckExecution("A") == nil {
		t.Fatal("captured A must never recompute")
	}
	if frozen.CheckReuse("A", uuid.New()) == nil {
		t.Fatal("captured A must keep its exact artifact")
	}
	if err := frozen.CheckReuse("A", id); err != nil {
		t.Fatal(err)
	}
	if err := frozen.CheckExecution("B"); err != nil {
		t.Fatalf("missing B has no prior result to freeze: %v", err)
	}
	plan := resolver.Plan{Stages: []resolver.ResolvedStage{
		{StageID: "A", Action: resolver.Reuse, ReusableArtifactID: &id},
		{StageID: "B", Action: resolver.Execute},
	}}
	if _, err := resolver.VerifyIntervention(plan, frozen.FrozenStageIDs, frozen.BaselineArtifactIDs); err != nil {
		t.Fatalf("partial continuation failed before missing-prefix execution: %v", err)
	}
	plan.Stages[0].Action = resolver.Execute
	if _, err := resolver.VerifyIntervention(plan, frozen.FrozenStageIDs, frozen.BaselineArtifactIDs); err == nil {
		t.Fatal("captured baseline A must still reject cold-cache execution")
	}
}
