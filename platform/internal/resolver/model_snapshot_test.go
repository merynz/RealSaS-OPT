package resolver

import (
	"context"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
)

func TestLegacyResolverSemanticsRemainUnchangedWithoutModelSnapshot(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")}}
	versions := resolverVersions(g, hex64("1"))

	legacy, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions)
	if err != nil {
		t.Fatal(err)
	}
	extended, err := ResolveWithExternalAndModels(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(legacy.Stages) != len(extended.Stages) {
		t.Fatalf("legacy stage count drifted: %d != %d", len(legacy.Stages), len(extended.Stages))
	}
	for i := range legacy.Stages {
		if legacy.Stages[i].ExpectedSemanticSHA256 != extended.Stages[i].ExpectedSemanticSHA256 {
			t.Fatalf("legacy semantic identity drifted at %s", legacy.Stages[i].StageID)
		}
	}
}

func TestExactModelSnapshotControlsQualifiedReuse(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")}}
	versions := resolverVersions(g, hex64("1"))
	modelA := []domain.ArtifactInputIdentity{{Role: "model:IRIS", ArtifactType: "RealSaS.ModelReleaseSelection", SemanticSHA256: hex64("b")}}
	modelB := []domain.ArtifactInputIdentity{{Role: "model:IRIS", ArtifactType: "RealSaS.ModelReleaseSelection", SemanticSHA256: hex64("c")}}

	baseline, err := ResolveWithExternalAndModels(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, modelA)
	if err != nil {
		t.Fatal(err)
	}
	catalog.admit(baseline)

	sameModel, err := ResolveWithExternalAndModels(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, modelA)
	if err != nil {
		t.Fatal(err)
	}
	if len(sameModel.ExecuteStageIDs()) != 0 {
		t.Fatalf("same exact model snapshot did not reuse qualified artifacts: %v", sameModel.ExecuteStageIDs())
	}

	changed, err := ResolveWithExternalAndModels(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, modelB)
	if err != nil {
		t.Fatal(err)
	}
	if len(changed.ExecuteStageIDs()) != len(changed.Stages) {
		t.Fatalf("changed exact model snapshot reused stale artifacts: execute=%d total=%d", len(changed.ExecuteStageIDs()), len(changed.Stages))
	}
}

func TestModelSemanticInputOrderingIsCanonical(t *testing.T) {
	g := resolverGraph(t)
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")}}
	versions := resolverVersions(g, hex64("1"))
	a := domain.ArtifactInputIdentity{Role: "model:IRIS", ArtifactType: "RealSaS.ModelReleaseSelection", SemanticSHA256: hex64("b")}
	b := domain.ArtifactInputIdentity{Role: "model:TESSA", ArtifactType: "RealSaS.ModelReleaseSelection", SemanticSHA256: hex64("c")}

	forward, err := ResolveWithExternalAndModels(context.Background(), g, newMemoryCatalog(), "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, []domain.ArtifactInputIdentity{a, b})
	if err != nil {
		t.Fatal(err)
	}
	reverse, err := ResolveWithExternalAndModels(context.Background(), g, newMemoryCatalog(), "46_PRODUCT_CLOSURE_SEAL", root, versions, nil, []domain.ArtifactInputIdentity{b, a})
	if err != nil {
		t.Fatal(err)
	}
	for i := range forward.Stages {
		if forward.Stages[i].ExpectedSemanticSHA256 != reverse.Stages[i].ExpectedSemanticSHA256 {
			t.Fatalf("model input order changed semantic identity at %s", forward.Stages[i].StageID)
		}
	}
}
