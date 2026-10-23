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
