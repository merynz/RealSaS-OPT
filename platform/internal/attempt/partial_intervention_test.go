package attempt

import (
	"github.com/google/uuid"
	"testing"
)

func TestPartialParentPreservesActualArtifactsAndMayComputeMissingPrefix(t *testing.T) {
	id := uuid.New()
	frozen := FrozenIntervention{FrozenStageIDs: []string{"A", "B"}, BaselineArtifactIDs: map[string]uuid.UUID{"A": id}}
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
}
