package release

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func modelRequirementGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	raw, err := json.Marshal(stagegraph.Snapshot{
		StageCount: 1,
		Stages:     []stagegraph.Stage{{Ordinal: 1, ID: "PRODUCT", Adapter: "product.py"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func productManifestForModels() Manifest {
	return Manifest{
		Name:    "product-model-binding-test",
		Purpose: PurposeProduct,
		Stages: []StageBinding{{
			Ordinal:              1,
			StageID:              "PRODUCT",
			ImplementationSHA256: strings.Repeat("a", 64),
			PolicySHA256:         strings.Repeat("b", 64),
			SemanticParameters:   map[string]any{},
		}},
	}
}

func TestLegacyManifestOmitsModelRequirements(t *testing.T) {
	m := productManifestForModels().normalized()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "model_requirements") {
		t.Fatalf("legacy manifest identity changed: %s", raw)
	}
	if err := m.Validate(modelRequirementGraph(t)); err != nil {
		t.Fatal(err)
	}
}

func TestProductModelRequirementsAreDeterministicAndProductionOnly(t *testing.T) {
	g := modelRequirementGraph(t)
	m := productManifestForModels()
	m.ModelRequirements = []ModelRequirement{
		{Family: "skin", ArchitectureID: "skin-v1", Channel: "PRODUCTION"},
		{Family: "iris", ArchitectureID: "iris-vnext", Channel: "PRODUCTION"},
	}
	normalized := m.normalized()
	if normalized.ModelRequirements[0].Family != "iris" || normalized.ModelRequirements[1].Family != "skin" {
		t.Fatalf("requirements are not canonicalized: %+v", normalized.ModelRequirements)
	}
	if err := normalized.Validate(g); err != nil {
		t.Fatal(err)
	}

	bad := m
	bad.ModelRequirements = []ModelRequirement{{Family: "iris", ArchitectureID: "iris-vnext", Channel: "CANDIDATE"}}
	if err := bad.Validate(g); err == nil || !strings.Contains(err.Error(), "PRODUCTION channel") {
		t.Fatalf("product candidate channel admitted: %v", err)
	}
}

func TestDuplicateModelRequirementFamilyFailsClosed(t *testing.T) {
	m := productManifestForModels()
	m.ModelRequirements = []ModelRequirement{
		{Family: "iris", ArchitectureID: "iris-vnext", Channel: "PRODUCTION"},
		{Family: "iris", ArchitectureID: "iris-other", Channel: "PRODUCTION"},
	}
	if err := m.Validate(modelRequirementGraph(t)); err == nil || !strings.Contains(err.Error(), "duplicate model requirement") {
		t.Fatalf("duplicate family admitted: %v", err)
	}
}
