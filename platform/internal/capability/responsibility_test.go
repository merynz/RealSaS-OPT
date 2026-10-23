package capability

import (
	"context"
	"reflect"
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
	if shape.ResponsibilityOwnerClass != ImplementationModel || shape.OwnerComponent != "TESSA" || shape.ModelFamily != "TESSA" {
		t.Fatalf("shape model terrain=%+v", shape)
	}
	iris := byID["shape.shared_scene_evidence"]
	if iris.OwnerComponent != "IRIS" || iris.ModelFamily != "IRIS" {
		t.Fatalf("iris terrain=%+v", iris)
	}
}

func TestRequiredModelFamiliesAreDerivedFromActiveTerrain(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	families, err := RequiredModelFamilies(registry)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"IRIS", "TESSA"}; !reflect.DeepEqual(families, want) {
		t.Fatalf("families=%v want=%v", families, want)
	}
}

func TestRequiredModelFamiliesIncludeAxisWhenRigIsActive(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	families, err := RequiredModelFamilies(registry)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"AXIS", "IRIS", "TESSA"}; !reflect.DeepEqual(families, want) {
		t.Fatalf("families=%v want=%v", families, want)
	}
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]ResponsibilityTerrainNode{}
	for _, row := range terrain {
		byID[row.ResponsibilityID] = row
	}
	axis := byID["rig.structure_proposal"]
	if axis.RoleID != ProductRoleRig || axis.ModelFamily != "AXIS" || axis.AuthorityKind != "PROPOSAL" {
		t.Fatalf("axis terrain=%+v", axis)
	}
	if _, ok := byID["rig.frame_observability_qualification"]; ok {
		t.Fatal("post-bind frame observability still appears as a Rig responsibility")
	}
}

func TestResponsibilityLedgerSeparatesAxisMiraProposalsFromCompilerAuthority(t *testing.T) {
	ledger, err := loadResponsibilityLedger()
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]ResponsibilityDefinition{}
	for _, row := range ledger.Responsibilities {
		byID[row.ID] = row
	}
	axis := byID["rig.structure_proposal"]
	if axis.ExecutionOwnerClass != ImplementationModel || axis.OwnerComponent != "AXIS" || axis.ModelFamily != "AXIS" || axis.AuthorityKind != "PROPOSAL" {
		t.Fatalf("axis responsibility=%+v", axis)
	}
	rigMint := byID["rig.canonical_identity_mint"]
	if rigMint.ExecutionOwnerClass != ImplementationCompiler || rigMint.OwnerComponent != "COMPILER" || rigMint.AuthorityKind != "CANONICAL_AUTHORITY" {
		t.Fatalf("rig mint responsibility=%+v", rigMint)
	}
	if _, ok := byID["rig.frame_observability_qualification"]; ok {
		t.Fatal("legacy Rig frame observability responsibility still exists")
	}
	postBind := byID["mechanics.post_bind_frame_observability_qualification"]
	if postBind.RoleID != ProductRoleMechanics || postBind.ExecutionOwnerClass != ImplementationCompiler || postBind.AuthorityKind != "QUALIFICATION" {
		t.Fatalf("post-bind observability responsibility=%+v", postBind)
	}
	mira := byID["skin.influence_proposal"]
	if mira.ExecutionOwnerClass != ImplementationModel || mira.OwnerComponent != "MIRA" || mira.ModelFamily != "MIRA" || mira.AuthorityKind != "PROPOSAL" {
		t.Fatalf("mira responsibility=%+v", mira)
	}
	skinMint := byID["skin.canonical_influence_mint"]
	if skinMint.ExecutionOwnerClass != ImplementationCompiler || skinMint.OwnerComponent != "COMPILER" || skinMint.AuthorityKind != "CANONICAL_AUTHORITY" {
		t.Fatalf("skin mint responsibility=%+v", skinMint)
	}
}

func TestResponsibilityTerrainRejectsMissingRequiredResponsibility(t *testing.T) {
	inputRows, err := ProductInputProvider{}.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	shapeRows, err := ProductShapeProvider{}.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	metadata := map[string]any{}
	for key, value := range shapeRows[0].Metadata {
		metadata[key] = value
	}
	metadata["responsibility_ids"] = []string{"shape.topology_proposal"}
	shapeRows[0].Metadata = metadata
	registry, err := NewRegistry(append(inputRows, shapeRows...))
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
