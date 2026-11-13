package capability

import (
	"context"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func rssUnifiedTestRegistry(t *testing.T) *Registry {
	t.Helper()
	registry, err := Build(
		context.Background(),
		RSSvNextInputCandidateProvider{},
		RSSvNextShapeCandidateProvider{},
		RSSvNextMeshCandidateProvider{},
		RSSvNextRigCandidateProvider{},
		RSSvNextSkinCandidateProvider{},
	)
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func TestProductModelRequirementsFollowOneUnifiedAttemptNotFourWorkerReleases(t *testing.T) {
	registry := rssUnifiedTestRegistry(t)
	rows, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("RSS model requirement=%d want=1: %+v", len(rows), rows)
	}
	rss := rows[0]
	if rss.Family != RSSvNextModelFamily || rss.ArchitectureID != RSSvNextArchitectureID ||
		rss.RoleID != ProductRoleShape || rss.CapabilityID != ProductShapeCapabilityID {
		t.Fatalf("one Shape execution owns the learned release: %+v", rss)
	}
	families, err := RequiredModelFamilies(registry)
	if err != nil || len(families) != 1 || families[0] != RSSvNextModelFamily {
		t.Fatalf("RSS learned family ownership=%v / %v", families, err)
	}
}

func TestProductModelRequirementsExactlyOneForFullNineRoleCatalog(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	rows, err := ProductModelRequirements(registry)
	if err != nil || len(rows) != 1 ||
		rows[0].Family != RSSvNextModelFamily ||
		rows[0].RoleID != ProductRoleShape {
		t.Fatalf("nine-role RSS model requirements=%+v / %v", rows, err)
	}
	terrain, err := ResponsibilityTerrain(registry)
	if err != nil {
		t.Fatal(err)
	}
	claimed := map[string]bool{}
	for _, item := range terrain {
		if item.ModelFamily == RSSvNextModelFamily {
			claimed[item.ResponsibilityID] = true
			if item.ModelExecutionCapabilityID != "product.shape" {
				t.Fatalf("RSS expert must originate from Shape release: %+v", item)
			}
		}
	}
	for _, id := range []string{
		"shape.shared_scene_evidence", "shape.topology_proposal",
		"rig.structure_proposal", "skin.influence_proposal",
	} {
		if !claimed[id] {
			t.Fatalf("missing unified source ownership for %s", id)
		}
	}
}

func TestFullProductCatalogEngineReleaseRejectsIndependentAxisMiraIrisTessa(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	valid := []release.ModelRequirement{{
		Family: RSSvNextModelFamily, ArchitectureID: RSSvNextArchitectureID,
		Channel: "PRODUCTION",
	}}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, valid); err != nil {
		t.Fatal(err)
	}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, nil); err == nil {
		t.Fatal("accepted empty model set")
	}
	legacy := append(append([]release.ModelRequirement{}, valid...), release.ModelRequirement{
		Family: "MIRA", ArchitectureID: "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0", Channel: "PRODUCTION",
	})
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, legacy); err == nil {
		t.Fatal("admitted independent MIRA model with RSS unified checkpoint")
	}
	wrong := []release.ModelRequirement{{
		Family: RSSvNextModelFamily, ArchitectureID: "wrong",
		Channel: "PRODUCTION",
	}}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrong); err == nil {
		t.Fatal("accepted unified architecture drift")
	}
}

func TestEngineReleaseModelRequirementsRejectArchitectureAndChannelDrift(t *testing.T) {
	registry, err := Build(context.Background(), RSSvNextInputCandidateProvider{}, RSSvNextShapeCandidateProvider{})
	if err != nil {
		t.Fatal(err)
	}
	valid := []release.ModelRequirement{{
		Family: RSSvNextModelFamily, ArchitectureID: RSSvNextArchitectureID,
		Channel: "PRODUCTION",
	}}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, valid); err != nil {
		t.Fatal(err)
	}
	research := []release.ModelRequirement{{
		Family: RSSvNextModelFamily, ArchitectureID: RSSvNextArchitectureID,
		Channel: "RESEARCH",
	}}
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeResearch, research); err != nil {
		t.Fatal(err)
	}
	wrong := append([]release.ModelRequirement(nil), valid...)
	wrong[0].ArchitectureID = "bad"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrong); err == nil {
		t.Fatal("architecture drift accepted")
	}
	wrong = append([]release.ModelRequirement(nil), valid...)
	wrong[0].Channel = "RESEARCH"
	if err := validateEngineReleaseModelRequirements(registry, release.PurposeProduct, wrong); err == nil {
		t.Fatal("product release accepted research channel")
	}
}

func TestProductShapeMeshRetainSevenCASChildrenAndOneCanonicalMesh(t *testing.T) {
	shapeRows, err := (RSSvNextShapeCandidateProvider{}).Capabilities(context.Background())
	if err != nil || len(shapeRows) != 1 {
		t.Fatalf("Shape descriptor=%v / %d", err, len(shapeRows))
	}
	shapeMeta := shapeRows[0].Metadata
	if shapeMeta["fanout_contract"] != "RealSaS.RSSvNext.ProductEvidenceFanout.v1" ||
		shapeMeta["neural_scene_survives_mesh_abstain"] != true ||
		shapeMeta["independent_iris_tessa_forbidden"] != true {
		t.Fatalf("missing one-inference retained fanout: %+v", shapeMeta)
	}
	aux, ok := shapeMeta["auxiliary_outputs"].([]any)
	if !ok || len(aux) != 7 {
		t.Fatalf("expected seven independently retained Shape CAS children %+v", aux)
	}
	meshRows, err := (RSSvNextMeshCandidateProvider{}).Capabilities(context.Background())
	if err != nil || len(meshRows) != 1 {
		t.Fatalf("mesh descriptor=%v / %d", err, len(meshRows))
	}
	meshMeta := meshRows[0].Metadata
	if meshMeta["output_artifact_type"] != "RealSaS.CanonicalMeshIR" ||
		meshMeta["canonical_mesh_required_for_product_execution"] != true ||
		meshMeta["input_shape_artifact_type"] != "RealSaS.RSSvNextInferenceEvidence" ||
		meshMeta["independent_model_inference_forbidden"] != true {
		t.Fatalf("canonical product mesh is not source-gated: %+v", meshMeta)
	}
	if x, ok := meshMeta["two_view_static_policy_sha256"].(string); !ok || len(x) != 64 {
		t.Fatalf("source-raster policy must be content pinned: %+v", meshMeta)
	}
	// The research adapter still exists as an explicit non-active legacy
	// implementation; it must not enter the current ModelRelease catalog.
	for _, provider := range ProductProviders() {
		if provider == nil {
			t.Fatal("nil product provider")
		}
	}
}
