package capability

import (
	"context"
	"strings"
	"testing"
)

func TestRSSvNextTwoViewInputAndSingleModelShapeFormOneCandidateRoleGraph(t *testing.T) {
	registry, err := Build(
		context.Background(),
		RSSvNextInputCandidateProvider{},
		RSSvNextShapeCandidateProvider{},
	)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := registry.Resolve(ProductShapeCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].ID != ProductInputCapabilityID ||
		resolved[1].ID != ProductShapeCapabilityID {
		t.Fatalf("RSS product dependency order drift: %+v", resolved)
	}

	input, ok := registry.Get(ProductInputCapabilityID)
	if !ok || input.Metadata["executor_contract"] != RSSvNextInputContractV2 ||
		input.Metadata["source_copy_policy"] != "BYTE_IDENTICAL_CAS_REUSE" ||
		input.Metadata["model_inference_in_input_forbidden"] != true ||
		input.ProductPassAuthority {
		t.Fatalf("RSS Input drift: %+v", input)
	}
	shape, ok := registry.Get(ProductShapeCapabilityID)
	if !ok || shape.Metadata["executor_contract"] != RSSvNextShapeContractV3 ||
		shape.Metadata["input_artifact_type"] != input.Metadata["output_artifact_type"] ||
		shape.Metadata["input_schema_version"] != input.Metadata["output_schema_version"] ||
		shape.Metadata["neural_scene_survives_mesh_abstain"] != true ||
		shape.Metadata["independent_iris_tessa_forbidden"] != true ||
		shape.ProductPassAuthority {
		t.Fatalf("RSS single-checkpoint Shape drift: %+v", shape)
	}
	requirements, err := descriptorRequiredModels(shape)
	if err != nil || len(requirements) != 1 {
		t.Fatalf("Expected exactly one unified RSS model: %+v, %v", requirements, err)
	}
	if requirements[0].Family != RSSvNextModelFamily ||
		requirements[0].ArchitectureID != RSSvNextArchitectureID {
		t.Fatalf("RSS model ABI drift: %+v", requirements[0])
	}
	aux, ok := shape.Metadata["auxiliary_outputs"].([]any)
	if !ok || len(aux) != 7 {
		t.Fatalf("RSS must preserve seven independently consumable CAS outputs: %+v", aux)
	}
	wanted := map[string]bool{
		"original_source":         true,
		"normalized_model_source": true,
		"neural_scene":            true,
		"inference_receipt":       true,
		"unified_model_config":    true,
		"embedded_mira_decoder":  true,
		"rss_evidence_fanout":     true,
	}
	for _, raw := range aux {
		row, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("RSS auxiliary artifact type drift: %+v", raw)
		}
		role, ok := row["role"].(string)
		if !ok || !wanted[role] {
			t.Fatalf("RSS auxiliary duplicate/unknown: %+v", row)
		}
		delete(wanted, role)
	}
	if len(wanted) != 0 {
		t.Fatalf("RSS retained evidence not fully declared: %+v", wanted)
	}
	graph, err := ProductRoleGraph(registry)
	if err != nil || len(graph) != 2 {
		t.Fatalf("RSS product roles not resolvable: %+v / %v", graph, err)
	}
	// The old responsibility ledger currently pins IRIS and TESSA families.
	// Do not falsely report this candidate as an admitted production release.
	_, err = ProductModelRequirements(registry)
	if err == nil || !strings.Contains(err.Error(), "PRODUCT_MODEL_REQUIREMENT_") {
		t.Fatalf("Old responsibility ledger unexpectedly admitted new RSS release: %v", err)
	}
}

func TestRSSvNextProvidersAreAlternativesNotParallelIndependentProductModels(t *testing.T) {
	_, err := Build(
		context.Background(),
		ProductInputProvider{}, RSSvNextInputCandidateProvider{},
	)
	if err == nil || !strings.Contains(err.Error(), "duplicate capability id") {
		t.Fatalf("Historical and RSS product.input both active: %v", err)
	}
	_, err = Build(
		context.Background(),
		ProductShapeProvider{}, RSSvNextShapeCandidateProvider{},
		RSSvNextInputCandidateProvider{},
	)
	if err == nil || !strings.Contains(err.Error(), "duplicate capability id") {
		t.Fatalf("Historical and RSS product.shape both active: %v", err)
	}
}

func TestRSSvNextCandidateMeshConsumesExactUnifiedShapeNotOldEightViewStages(t *testing.T) {
	registry, err := Build(
		context.Background(),
		RSSvNextInputCandidateProvider{},
		RSSvNextShapeCandidateProvider{},
		RSSvNextMeshCandidateProvider{},
	)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := registry.Resolve(ProductMeshCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 3 || resolved[0].ID != ProductInputCapabilityID ||
		resolved[1].ID != ProductShapeCapabilityID ||
		resolved[2].ID != ProductMeshCapabilityID {
		t.Fatalf("RSS Input→Shape→Mesh dependency drift: %+v", resolved)
	}
	mesh, ok := registry.Get(ProductMeshCapabilityID)
	if !ok || mesh.Metadata["executor_contract"] != RSSvNextMeshContractV2 ||
		mesh.Metadata["input_source_artifact_type"] != "RealSaS.RSSvNextTwoViewSource" ||
		mesh.Metadata["input_fanout_artifact_type"] != "RealSaS.RSSvNextProductEvidenceFanout" ||
		mesh.Metadata["output_artifact_type"] != "RealSaS.RSSvNextComputedCandidateSurface" ||
		mesh.Metadata["requires_shape_parent_binding"] != true ||
		mesh.Metadata["independent_model_inference_forbidden"] != true ||
		mesh.Metadata["learned_evidence_retained_by_reference"] != true ||
		mesh.Metadata["canonical_mesh_authority"] != false ||
		mesh.ProductPassAuthority {
		t.Fatalf("RSS Mesh v2 candidate contract drift: %+v", mesh)
	}
	requirements, err := descriptorRequiredModels(mesh)
	if err != nil || len(requirements) != 0 {
		t.Fatalf("RSS mesh must not bring its own independent checkpoints: %+v / %v", requirements, err)
	}
	// Active platform release still awaits the source-aware responsibility
	// ledger and downstream canonical geometry/mechanics admission switch.
	_, err = ProductModelRequirements(registry)
	if err == nil || !strings.Contains(err.Error(), "PRODUCT_MODEL_REQUIREMENT_") {
		t.Fatalf("RSS candidate wrongly admitted against old ledger: %v", err)
	}

	_, err = Build(
		context.Background(),
		ProductMeshProvider{}, RSSvNextMeshCandidateProvider{},
		RSSvNextInputCandidateProvider{}, RSSvNextShapeCandidateProvider{},
	)
	if err == nil || !strings.Contains(err.Error(), "duplicate capability id") {
		t.Fatalf("Old and new Mesh simultaneously registered: %v", err)
	}
}
