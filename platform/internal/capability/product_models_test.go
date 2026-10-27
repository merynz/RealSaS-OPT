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
	if axis.ArchitectureID != "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1" || axis.RoleID != ProductRoleRig {
		t.Fatalf("axis requirement=%+v", axis)
	}
	mira := byFamily["MIRA"]
	if mira.ArchitectureID != "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5" || mira.RoleID != ProductRoleSkin {
		t.Fatalf("mira requirement=%+v", mira)
	}
	if byFamily["IRIS"].RoleID != ProductRoleShape || byFamily["TESSA"].RoleID != ProductRoleShape {
		t.Fatalf("shape requirements=%+v", byFamily)
	}
}

func TestFullProductCatalogEngineReleaseRequiresAxisAndMira(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	valid := []release.ModelRequirement{
		{Family: "AXIS", ArchitectureID: "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1", Channel: "PRODUCTION"},
		{Family: "IRIS", ArchitectureID: "RealSaS.IRIS.VNextEvidenceModel.v3", Channel: "PRODUCTION"},
		{Family: "MIRA", ArchitectureID: "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5", Channel: "PRODUCTION"},
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
