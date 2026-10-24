package release

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func releaseIdentityTestGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	stages := []stagegraph.Stage{{Ordinal: 1, ID: "SOURCE"}}
	raw, err := json.Marshal(stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: len(stages),
		Stages:     stages,
	})
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return graph
}

func releaseIdentityManifest(implementation string) Manifest {
	return Manifest{
		ContractVersion: "RealSaS.EngineReleaseManifest.v2",
		Name:            "capability-identity-test",
		Purpose:         PurposeResearch,
		Stages: []StageBinding{{
			Ordinal:              1,
			StageID:              "SOURCE",
			ImplementationSHA256: strings.Repeat("a", 64),
			PolicySHA256:         strings.Repeat("b", 64),
			SemanticParameters:   map[string]any{},
		}},
		CapabilityImplementations: map[string]string{
			"product.input": implementation,
		},
	}
}

func TestV2ReleaseIdentityChangesWhenProductImplementationChanges(t *testing.T) {
	graph := releaseIdentityTestGraph(t)
	first := releaseIdentityManifest(strings.Repeat("c", 64))
	second := releaseIdentityManifest(strings.Repeat("d", 64))
	firstSHA, err := first.SHA256(graph)
	if err != nil {
		t.Fatal(err)
	}
	secondSHA, err := second.SHA256(graph)
	if err != nil {
		t.Fatal(err)
	}
	if firstSHA == secondSHA {
		t.Fatal("product implementation change must change EngineRelease identity")
	}
}

func TestV2ReleaseIdentityRequiresProductImplementationBinding(t *testing.T) {
	graph := releaseIdentityTestGraph(t)
	manifest := releaseIdentityManifest(strings.Repeat("c", 64))
	manifest.CapabilityImplementations = nil
	if _, err := manifest.SHA256(graph); err == nil {
		t.Fatal("V2 EngineRelease without product implementation binding must fail closed")
	}
}

func TestV2ReleaseIdentityCapabilityMapOrderIsNotAuthority(t *testing.T) {
	graph := releaseIdentityTestGraph(t)
	first := releaseIdentityManifest(strings.Repeat("c", 64))
	first.CapabilityImplementations["product.shape"] = strings.Repeat("d", 64)
	second := releaseIdentityManifest(strings.Repeat("c", 64))
	second.CapabilityImplementations = map[string]string{
		"product.shape": strings.Repeat("d", 64),
		"product.input": strings.Repeat("c", 64),
	}
	firstSHA, err := first.SHA256(graph)
	if err != nil {
		t.Fatal(err)
	}
	secondSHA, err := second.SHA256(graph)
	if err != nil {
		t.Fatal(err)
	}
	if firstSHA != secondSHA {
		t.Fatal("map insertion order must not change EngineRelease identity")
	}
}
