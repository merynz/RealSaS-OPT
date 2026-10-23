package capability

import (
	"context"
	"testing"
)

func TestResponsibilityTerrainProjectsActiveInputAndShape(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(terrain) != 3 {
		t.Fatalf("terrain=%d want=3", len(terrain))
	}
	byID := map[string]ResponsibilityTerrainNode{}
	for _, row := range terrain {
		byID[row.ResponsibilityID] = row
	}
	input := byID["input.model_observation_lowering"]
	if input.RoleID != ProductRoleInput || input.DataReduction != "BOUNDED_LOSSY" || !input.SourceTruthRetained {
		t.Fatalf("input terrain=%+v", input)
	}
	shape := byID["shape.topology_proposal"]
	if shape.RoleID != ProductRoleShape || shape.AuthorityKind != "PROPOSAL" || shape.ImplementationClass != ImplementationModel {
		t.Fatalf("shape terrain=%+v", shape)
	}
}

func TestResponsibilityTerrainRejectsMissingRequiredResponsibility(t *testing.T) {
	rows, err := ProductShapeProvider{}.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	metadata := map[string]any{}
	for key, value := range rows[0].Metadata {
		metadata[key] = value
	}
	metadata["responsibility_ids"] = []string{"shape.topology_proposal"}
	rows[0].Metadata = metadata
	registry, err := NewRegistry(rows)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ResponsibilityTerrain(registry); err == nil {
		t.Fatal("missing required responsibility accepted")
	}
}

func TestResponsibilityTerrainRejectsResponsibilityOwnedByWrongRole(t *testing.T) {
	rows, err := ProductInputProvider{}.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	metadata := map[string]any{}
	for key, value := range rows[0].Metadata {
		metadata[key] = value
	}
	metadata["responsibility_ids"] = []string{"shape.topology_proposal"}
	rows[0].Metadata = metadata
	registry, err := NewRegistry(rows)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ResponsibilityTerrain(registry); err == nil {
		t.Fatal("wrong-role responsibility accepted")
	}
}

func TestResponsibilityLedgerForbidsLossyTruthReplacement(t *testing.T) {
	ledger, err := loadResponsibilityLedger()
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range ledger.Responsibilities {
		if row.DataReduction == "BOUNDED_LOSSY" || row.DataReduction == "GENERATIVE_PROPOSAL" {
			if !row.SourceTruthRetained {
				t.Fatalf("lossy responsibility replaces source truth: %s", row.ID)
			}
		}
	}
}
