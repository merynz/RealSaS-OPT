package capability

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func partialReleaseTestGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	raw, err := json.Marshal(stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: 1,
		Stages: []stagegraph.Stage{{
			Ordinal: 1,
			ID:      "05_CAMERA_CONTRACT_SOLVED",
			Title:   "camera",
			Group:   "observation",
			Adapter: "x:y",
			Policy:  stagegraph.Policy{FailClosed: true},
		}},
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

func TestStageOnlyResearchReleaseRetainsFullRSSProductDescriptorGraph(t *testing.T) {
	registry, err := BuildEngineReleaseRegistry(context.Background(), partialReleaseTestGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	// The 2026 RSS source catalog is stage-independent: it is legal for
	// a RESEARCH release to pin its nine descriptors. Admission does not
	// imply a successful execution, trained checkpoint or ProductRevision.
	for _, capabilityID := range []string{
		ProductInputCapabilityID, ProductShapeCapabilityID,
		ProductMeshCapabilityID, ProductRigCapabilityID,
		ProductSkinCapabilityID, ProductMechanicsCapabilityID,
		ProductGraphicsCapabilityID, ProductMotionCapabilityID,
		ProductProofCapabilityID,
	} {
		row, ok := registry.Get(capabilityID)
		if !ok {
			t.Fatalf("research source catalog lost RSS role %s", capabilityID)
		}
		if row.ProductPassAuthority && capabilityID != ProductProofCapabilityID {
			t.Fatalf("non-Proof role claimed ProductPASS: %s", capabilityID)
		}
	}
	expected := []string{
		ProductInputCapabilityID, ProductShapeCapabilityID,
		ProductMeshCapabilityID, ProductRigCapabilityID,
		ProductSkinCapabilityID, ProductMechanicsCapabilityID,
		ProductGraphicsCapabilityID, ProductMotionCapabilityID,
		ProductProofCapabilityID,
	}
	for _, role := range expected {
		closure, err := registry.Resolve(role)
		if err != nil || len(closure) == 0 || closure[len(closure)-1].ID != role {
			t.Fatalf("role graph unresolved: %s: %+v / %v", role, closure, err)
		}
	}
	modelRequirements, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(modelRequirements) != 1 ||
		modelRequirements[0].Family != RSSvNextModelFamily ||
		modelRequirements[0].ArchitectureID != RSSvNextArchitectureID ||
		modelRequirements[0].CapabilityID != ProductShapeCapabilityID {
		t.Fatalf("research release selected independent specialist releases: %+v", modelRequirements)
	}
}

func TestModuleVersionIdentityIsMintedFromPlatformDescriptor(t *testing.T) {
	rows, err := (ProductInputProvider{}).Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("input provider descriptor count=%d", len(rows))
	}
	descriptor := rows[0]
	implementation := ModuleImplementationIdentity{
		ImplementationSHA256: strings.Repeat("a", 64),
	}
	version, err := mintModuleVersionIdentity(descriptor, implementation)
	if err != nil {
		t.Fatal(err)
	}
	expectedPolicy, err := modulePolicySHA256(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	expectedParameters, err := semantic.JSONSHA256(descriptor.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	if version.ImplementationSHA256 != implementation.ImplementationSHA256 {
		t.Fatalf("implementation identity drift=%s", version.ImplementationSHA256)
	}
	if version.PolicySHA256 != expectedPolicy {
		t.Fatalf("policy identity not minted from descriptor=%s!=%s", version.PolicySHA256, expectedPolicy)
	}
	if version.ParametersSHA256 != expectedParameters {
		t.Fatalf("parameter identity not minted from descriptor=%s!=%s", version.ParametersSHA256, expectedParameters)
	}
	if err := version.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestModuleImplementationIdentityRejectsCallerSemanticHashesByConstruction(t *testing.T) {
	identity := ModuleImplementationIdentity{ImplementationSHA256: strings.Repeat("b", 64)}
	if err := identity.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := json.Marshal(identity); err != nil {
		t.Fatal(err)
	} else if raw, _ := json.Marshal(identity); strings.Contains(string(raw), "policy_sha256") || strings.Contains(string(raw), "parameters_sha256") {
		t.Fatalf("implementation request leaked Platform-owned semantic identity: %s", raw)
	}
}
