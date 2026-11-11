package capability

import (
	"context"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func TestProductModelRequirementsFollowActiveTerrain(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	rows, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("requirements=%d want=2", len(rows))
	}
	if rows[0].Family != "IRIS" || rows[0].ArchitectureID != "RealSaS.IRIS.VNextEvidenceModel.v3" || rows[0].RoleID != ProductRoleShape {
		t.Fatalf("iris requirement=%+v", rows[0])
	}
	if rows[1].Family != "TESSA" || rows[1].ArchitectureID != "RealSaS.TESSA.Structuralizer.v3" || rows[1].RoleID != ProductRoleShape {
		t.Fatalf("tessa requirement=%+v", rows[1])
	}
}

func TestProductModelRequirementsIncludeAxisAndMiraForFullCatalog(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	rows, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("requirements=%d want=4", len(rows))
	}
	byFamily := map[string]ProductModelRequirement{}
	for _, row := range rows {
		byFamily[row.Family] = row
	}
	axis := byFamily["AXIS"]
	if axis.ArchitectureID != "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0" || axis.RoleID != ProductRoleRig {
		t.Fatalf("axis requirement=%+v", axis)
	}
	mira := byFamily["MIRA"]
	if mira.ArchitectureID != "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0" || mira.RoleID != ProductRoleSkin {
		t.Fatalf("mira requirement=%+v", mira)
	}
	if byFamily["IRIS"].RoleID != ProductRoleShape || byFamily["TESSA"].RoleID != ProductRoleShape {
		t.Fatalf("shape requirements=%+v", byFamily)
	}
}

func TestFullProductCatalogEngineReleaseRequiresAxisAndMira(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	valid := []release.ModelRequirement{
		{Family: "AXIS", ArchitectureID: "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0", Channel: "PRODUCTION"},
		{Family: "IRIS", ArchitectureID: "RealSaS.IRIS.VNextEvidenceModel.v3", Channel: "PRODUCTION"},
		{Family: "MIRA", ArchitectureID: "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0", Channel: "PRODUCTION"},
		{Family: "TESSA", ArchitectureID: "RealSaS.TESSA.Structuralizer.v3", Channel: "PRODUCTION"},
	}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, valid); err != nil {
		t.Fatalf("full product model set rejected: %v", err)
	}
	missingAxis := []release.ModelRequirement{valid[1], valid[2], valid[3]}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, missingAxis); err == nil {
		t.Fatal("full product catalog accepted an EngineRelease without AXIS")
	}
	missingMira := []release.ModelRequirement{valid[0], valid[1], valid[3]}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, missingMira); err == nil {
		t.Fatal("full product catalog accepted an EngineRelease without MIRA")
	}
	wrongAxis := append([]release.ModelRequirement(nil), valid...)
	wrongAxis[0].ArchitectureID = "wrong"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrongAxis); err == nil {
		t.Fatal("active product.rig accepted AXIS architecture drift")
	}
	wrongMira := append([]release.ModelRequirement(nil), valid...)
	wrongMira[2].ArchitectureID = "wrong"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrongMira); err == nil {
		t.Fatal("active product.skin accepted MIRA architecture drift")
	}
}

func TestEngineReleaseModelRequirementsRejectArchitectureAndChannelDrift(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	valid := []release.ModelRequirement{
		{Family: "IRIS", ArchitectureID: "RealSaS.IRIS.VNextEvidenceModel.v3", Channel: "PRODUCTION"},
		{Family: "TESSA", ArchitectureID: "RealSaS.TESSA.Structuralizer.v3", Channel: "PRODUCTION"},
	}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, valid); err != nil {
		t.Fatal(err)
	}
	research := []release.ModelRequirement{
		{Family: "IRIS", ArchitectureID: "RealSaS.IRIS.VNextEvidenceModel.v3", Channel: "RESEARCH"},
		{Family: "TESSA", ArchitectureID: "RealSaS.TESSA.Structuralizer.v3", Channel: "CANDIDATE"},
	}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeResearch, research); err != nil {
		t.Fatal(err)
	}
	wrongArchitecture := append([]release.ModelRequirement(nil), valid...)
	wrongArchitecture[1].ArchitectureID = "wrong"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrongArchitecture); err == nil {
		t.Fatal("architecture drift accepted")
	}
	wrongChannel := append([]release.ModelRequirement(nil), valid...)
	wrongChannel[0].Channel = "RESEARCH"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrongChannel); err == nil {
		t.Fatal("product release accepted research model channel")
	}
}

func TestProductShapeMeshRetainIndependentCASSceneEvidence(t *testing.T) {
	shapeRows, err := (ProductShapeProvider{}).Capabilities(context.Background())
	if err != nil || len(shapeRows) != 1 {
		t.Fatalf("product.shape descriptor: %v / %d", err, len(shapeRows))
	}
	shapeMeta := shapeRows[0].Metadata
	if shapeMeta["evidence_fanout_contract"] != "RealSaS.ShapeEvidenceFanout.v1" ||
		shapeMeta["independent_scene_consumers_do_not_require_mesh"] != true ||
		shapeMeta["model_scene_evidence_retained_after_mesh_compile"] != true {
		t.Fatalf("product.shape dropped scene fan-out: %+v", shapeMeta)
	}
	aux, ok := shapeMeta["auxiliary_outputs"].([]any)
	if !ok || len(aux) != 2 {
		t.Fatalf("expected exactly two additional immutable Shape CAS outputs: %+v", aux)
	}
	wanted := map[string]string{
		"iris_scene_evidence":   "RealSaS.IRISSceneEvidence/v1",
		"shape_evidence_fanout": "RealSaS.ShapeEvidenceFanout/v1",
	}
	for _, raw := range aux {
		output, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("untyped auxiliary output: %+v", raw)
		}
		role, ok := output["role"].(string)
		if !ok {
			t.Fatalf("auxiliary role is not a string: %+v", output)
		}
		got := output["artifact_type"].(string) + "/" + output["schema_version"].(string)
		if wanted[role] != got {
			t.Fatalf("unexpected Shape evidence output %s: %s", role, got)
		}
		delete(wanted, role)
	}
	if len(wanted) != 0 {
		t.Fatalf("missing fan-out artifacts %+v", wanted)
	}
	meshRows, err := (ProductMeshProvider{}).Capabilities(context.Background())
	if err != nil || len(meshRows) != 1 {
		t.Fatalf("product.mesh descriptor: %v / %d", err, len(meshRows))
	}
	meshMeta := meshRows[0].Metadata
	if meshMeta["requires_shape_evidence_fanout"] != true ||
		meshMeta["iris_scene_evidence_loaded_by_mesh"] != false ||
		meshMeta["canonical_mesh_replaces_neural_scene"] != false {
		t.Fatalf("product.mesh must verify the lightweight index without dropping/decoding parent scene: %+v", meshMeta)
	}
}
