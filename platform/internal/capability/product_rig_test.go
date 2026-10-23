package capability

import (
	"context"
	"strings"
	"testing"
)

func TestProductRigCapabilityUsesAXISProposalAndCompilerAuthority(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	descriptor, ok := registry.Get(ProductRigCapabilityID)
	if !ok {
		t.Fatal("product.rig capability missing")
	}
	if descriptor.Kind != KindModule || descriptor.OwnerModuleID != ProductRigCapabilityID {
		t.Fatalf("descriptor=%+v", descriptor)
	}
	if len(descriptor.Dependencies) != 2 || descriptor.Dependencies[0] != ProductMeshCapabilityID || descriptor.Dependencies[1] != StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED") {
		t.Fatalf("dependencies=%v", descriptor.Dependencies)
	}
	if descriptor.ProductPassAuthority {
		t.Fatal("Rig may not claim product-pass authority")
	}
	if len(descriptor.AllowedModes) != 1 || descriptor.AllowedModes[0] != ModeDeveloper {
		t.Fatalf("allowed modes=%v", descriptor.AllowedModes)
	}
	if descriptor.Metadata["implementation_class"] != "HYBRID" {
		t.Fatalf("implementation class=%v", descriptor.Metadata["implementation_class"])
	}
	if descriptor.Metadata["executor_contract"] != ProductRigContractV1 {
		t.Fatalf("metadata=%v", descriptor.Metadata)
	}
	if descriptor.Metadata["canonical_parent_role"] != string(ProductRoleMesh) {
		t.Fatalf("canonical Rig parent must remain Mesh: %v", descriptor.Metadata)
	}
	if descriptor.Metadata["axis_input_adapter_contract"] != ProductRigAXISInputAdapterV1 {
		t.Fatalf("AXIS adapter contract drift: %v", descriptor.Metadata)
	}
	if descriptor.Metadata["axis_input_adapter_is_canonical_authority"] != false || descriptor.Metadata["axis_input_adapter_replaceable"] != true {
		t.Fatalf("AXIS adapter authority/replaceability drift: %v", descriptor.Metadata)
	}
	if descriptor.Metadata["surface_dependency_is_compatibility_evidence_only"] != true {
		t.Fatal("Stage15 surface must remain compatibility evidence, not canonical Rig parent")
	}
	if descriptor.Metadata["output_artifact_type"] != "RealSaS.QualifiedSkeletonIR" || descriptor.Metadata["output_authority_class"] != "COMPILER_CANONICAL_RIG" {
		t.Fatalf("Rig authority output drift: %v", descriptor.Metadata)
	}
	if descriptor.Metadata["compiler_parent_reselection_allowed"] != false {
		t.Fatal("Compiler parent reselection must remain forbidden on AXIS hard-tree path")
	}
	if descriptor.Metadata["proposal_authority_artifact_emitted"] != false {
		t.Fatal("learned proposal may not be emitted as Rig authority")
	}
	if descriptor.Metadata["canonical_mesh_mutation_allowed"] != false {
		t.Fatal("Rig may not mutate canonical Mesh")
	}
	if descriptor.Metadata["post_bind_frame_observability_claimed"] != false || descriptor.Metadata["frame_observability_deferred_to_mechanics"] != true {
		t.Fatal("post-bind frame observability must remain a Mechanics responsibility")
	}
	models, err := descriptorRequiredModels(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Family != "AXIS" || models[0].ArchitectureID != ProductRigAXISArchitectureID {
		t.Fatalf("models=%+v", models)
	}

	if _, err := (Goal{Type: GoalDeveloperRun, Targets: []string{ProductRigCapabilityID}, PromotionPolicy: PromotionNever}).Validate(registry); err != nil {
		t.Fatalf("developer Rig goal rejected: %v", err)
	}
	_, err = (Goal{Type: GoalProductCompile, Targets: []string{ProductRigCapabilityID}, PromotionPolicy: PromotionQualifiedProductOnly}).Validate(registry)
	if err == nil || !strings.Contains(err.Error(), "not product-pass authority") {
		t.Fatalf("product promotion should fail closed, err=%v", err)
	}
}

func TestRigResponsibilitiesSeparatePrebindEvidenceFromPostbindMechanics(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		t.Fatal(err)
	}
	seenRigAdmission := false
	seenRigFinalObservability := false
	for _, row := range terrain {
		if row.RoleID != ProductRoleRig {
			continue
		}
		if row.ResponsibilityID == "rig.frame_evidence_admission" {
			seenRigAdmission = true
		}
		if row.ResponsibilityID == "rig.frame_observability_qualification" {
			seenRigFinalObservability = true
		}
	}
	if !seenRigAdmission || seenRigFinalObservability {
		t.Fatalf("rig frame responsibility boundary drift: admission=%v final=%v", seenRigAdmission, seenRigFinalObservability)
	}
}

func TestProductRigProviderCompilesStandaloneWithContext(t *testing.T) {
	rows, err := (ProductRigProvider{}).Capabilities(context.Background())
	if err != nil || len(rows) != 1 {
		t.Fatalf("provider rows=%+v err=%v", rows, err)
	}
}
