package resolver

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type memoryCatalog struct {
	rows map[string]uuid.UUID
}

func newMemoryCatalog() *memoryCatalog {
	return &memoryCatalog{rows: map[string]uuid.UUID{}}
}

func (m *memoryCatalog) FindQualified(_ context.Context, _, _, semanticSHA string) (uuid.UUID, bool, error) {
	id, ok := m.rows[semanticSHA]
	return id, ok, nil
}

func (m *memoryCatalog) admit(plan Plan) {
	for _, row := range plan.Stages {
		m.rows[row.ExpectedSemanticSHA256] = uuid.New()
	}
}

func resolverGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func resolverVersions(g *stagegraph.Graph, stage42Impl string) map[string]release.StageVersion {
	out := map[string]release.StageVersion{}
	for _, s := range g.Stages() {
		impl := hex64("1")
		if s.ID == "42_RUNTIME_PROJECTION_AND_CAA_BINDING" {
			impl = stage42Impl
		}
		out[s.ID] = release.StageVersion{
			ImplementationSHA256: impl,
			PolicySHA256:         hex64("2"),
			ParametersSHA256:     hex64("3"),
		}
	}
	return out
}

func TestIdenticalCompileReusesQualifiedStageResults(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{{
		Role:           "subject:source",
		Ordinal:        0,
		ArtifactType:   "RealSaS.SourceImage",
		SemanticSHA256: hex64("a"),
	}}
	first, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, resolverVersions(g, hex64("1")))
	if err != nil {
		t.Fatal(err)
	}
	if len(first.ExecuteStageIDs()) == 0 {
		t.Fatal("first compile must execute unresolved stages")
	}
	catalog.admit(first)
	second, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, resolverVersions(g, hex64("1")))
	if err != nil {
		t.Fatal(err)
	}
	if len(second.ExecuteStageIDs()) != 0 || len(second.ReusedStageIDs()) != len(first.Stages) {
		t.Fatalf("execute=%v reused=%d total=%d", second.ExecuteStageIDs(), len(second.ReusedStageIDs()), len(first.Stages))
	}
}

func TestStage42ChangeInvalidatesOnlyRuntimeDescendants(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", Ordinal: 0, ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")}}
	baseline, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, resolverVersions(g, hex64("1")))
	if err != nil {
		t.Fatal(err)
	}
	catalog.admit(baseline)
	changed, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, resolverVersions(g, hex64("4")))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"42_RUNTIME_PROJECTION_AND_CAA_BINDING",
		"43_RSS_MATERIALIZE_COMPACT",
		"44_NATIVE_PACKAGE_OPEN_PLAYBACK",
		"45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
		"46_PRODUCT_CLOSURE_SEAL",
	}
	if !same(changed.ExecuteStageIDs(), want) {
		t.Fatalf("execute=%v want=%v", changed.ExecuteStageIDs(), want)
	}
	for _, id := range changed.ReusedStageIDs() {
		if id == "10_IRIS_FIT" {
			return
		}
	}
	t.Fatal("IRIS fit must remain reusable")
}

func TestSkinChangePreservesIndependentAppearanceAndModels(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{{Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")}}
	versions := resolverVersions(g, hex64("1"))
	baseline, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions)
	if err != nil {
		t.Fatal(err)
	}
	catalog.admit(baseline)
	changedVersion := versions["32_SKIN_QUALIFIED"]
	changedVersion.ImplementationSHA256 = hex64("f")
	versions["32_SKIN_QUALIFIED"] = changedVersion
	changed, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions)
	if err != nil {
		t.Fatal(err)
	}
	executed := map[string]bool{}
	for _, id := range changed.ExecuteStageIDs() {
		executed[id] = true
	}
	for _, id := range []string{"23_COMPLETE_APPEARANCE_ASSET_BAKED", "19_STATIC_CANONICAL_MESH_QUALIFIED", "28_SKELETON_QUALIFIED", "10_IRIS_FIT"} {
		if executed[id] {
			t.Fatalf("independent stage invalidated: %s", id)
		}
	}
	if !executed["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"] {
		t.Fatal("dynamic mechanics must be requalified")
	}
}

func TestMotionManifestChangePreservesAllModelArtifacts(t *testing.T) {
	g := resolverGraph(t)
	catalog := newMemoryCatalog()
	root := []domain.ArtifactInputIdentity{
		{Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a")},
		{Role: "subject:manifest:motion", Ordinal: 1, ArtifactType: "RealSaS.ManifestSection", SemanticSHA256: hex64("b")},
	}
	versions := resolverVersions(g, hex64("1"))
	baseline, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions)
	if err != nil {
		t.Fatal(err)
	}
	catalog.admit(baseline)
	root[1].SemanticSHA256 = hex64("c")
	changed, err := Resolve(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions)
	if err != nil {
		t.Fatal(err)
	}
	if len(changed.ExecuteStageIDs()) == 0 {
		t.Fatal("motion change must execute downstream stages")
	}
	for _, id := range changed.ExecuteStageIDs() {
		stage, _ := g.Get(id)
		if stage.Ordinal < 39 {
			t.Fatalf("motion change invalidated upstream stage: %s", id)
		}
	}
}

func same(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func hex64(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
