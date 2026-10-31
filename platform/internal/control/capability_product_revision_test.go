package control

import (
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
)

func TestModularProductRevisionContractIsProofGatedAndCanonical(t *testing.T) {
	contract := modularProductRevisionContract()
	if contract.Name != "RealSaS.ProductRevision" || contract.Version != "v2" {
		t.Fatalf("contract identity=%s/%s", contract.Name, contract.Version)
	}
	if err := contract.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(contract.Roles) != 7 {
		t.Fatalf("revision role count=%d want=7", len(contract.Roles))
	}
	want := map[string][2]string{
		modularRevisionMeshRole:      {"RealSaS.CanonicalMeshIR", "v1"},
		modularRevisionRigRole:       {"RealSaS.CanonicalRigIR", "v2"},
		modularRevisionSkinRole:      {"RealSaS.CanonicalSkinIR", "v2"},
		modularRevisionMechanicsRole: {"RealSaS.CanonicalMechanicsIR", "v1"},
		modularRevisionGraphicsRole:  {"RealSaS.CanonicalGraphicsIR", "v1"},
		modularRevisionMotionRole:    {"RealSaS.CanonicalMotionIR", "v2"},
		modularRevisionProofRole:     {"RealSaS.ProductProofBundleIR", "v1"},
	}
	for _, role := range contract.Roles {
		expected, ok := want[role.Role]
		if !ok {
			t.Fatalf("unexpected revision role %s", role.Role)
		}
		if role.ArtifactType != expected[0] || role.SchemaVersion != expected[1] {
			t.Fatalf("role %s artifact=%s/%s want=%s/%s", role.Role, role.ArtifactType, role.SchemaVersion, expected[0], expected[1])
		}
		if role.QualificationType != "CAPABILITY_EXECUTION_PASS" || !role.Required {
			t.Fatalf("role %s qualification=%s required=%v", role.Role, role.QualificationType, role.Required)
		}
	}
	if contract.Metadata["terminal_product_pass_capability"] != capability.ProductProofCapabilityID {
		t.Fatalf("terminal Proof=%v", contract.Metadata["terminal_product_pass_capability"])
	}
	if contract.Metadata["legacy_stage46_required"] != false || contract.Metadata["legacy_runtime_package_required"] != false {
		t.Fatalf("legacy product authority leaked into modular revision: %+v", contract.Metadata)
	}
	if contract.Metadata["proof_owns_product_pass_decision"] != true || contract.Metadata["finalizer_owns_revision_seal"] != true {
		t.Fatalf("Proof/finalizer responsibility drift: %+v", contract.Metadata)
	}
}

func TestModularProductRevisionSourcesMatchShippingCapabilityRoles(t *testing.T) {
	want := map[string]string{
		"capability:product.mesh:canonical_mesh":           modularRevisionMeshRole,
		"capability:product.rig:canonical_rig":             modularRevisionRigRole,
		"capability:product.skin:canonical_skin":           modularRevisionSkinRole,
		"capability:product.mechanics:canonical_mechanics": modularRevisionMechanicsRole,
		"capability:product.graphics:canonical_graphics":   modularRevisionGraphicsRole,
		"capability:product.motion:canonical_motion":       modularRevisionMotionRole,
		"capability:product.proof:product_proof":           modularRevisionProofRole,
	}
	if len(modularRevisionSources) != len(want) {
		t.Fatalf("source count=%d want=%d", len(modularRevisionSources), len(want))
	}
	for _, source := range modularRevisionSources {
		if want[source.SourceRole] != source.RevisionRole {
			t.Fatalf("source binding drift: %+v", source)
		}
	}
}
