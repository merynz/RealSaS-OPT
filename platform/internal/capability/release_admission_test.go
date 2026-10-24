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

func TestPartialEngineReleaseAdmitsOnlyExecutableProductRoles(t *testing.T) {
	registry, err := BuildEngineReleaseRegistry(context.Background(), partialReleaseTestGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, capabilityID := range []string{ProductInputCapabilityID, ProductShapeCapabilityID} {
		if _, ok := registry.Get(capabilityID); !ok {
			t.Fatalf("partial release lost executable role %s", capabilityID)
		}
	}
	for _, capabilityID := range []string{ProductMeshCapabilityID, ProductRigCapabilityID} {
		if _, ok := registry.Get(capabilityID); ok {
			t.Fatalf("partial release advertised unavailable role %s", capabilityID)
		}
	}

	modelRequirements, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	families := map[string]bool{}
	for _, row := range modelRequirements {
		families[row.Family] = true
	}
	if !families["IRIS"] || !families["TESSA"] || len(families) != 2 {
		t.Fatalf("partial release model requirements=%+v", modelRequirements)
	}
	if families["AXIS"] {
		t.Fatalf("partial release required AXIS without admitted Rig: %+v", modelRequirements)
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
