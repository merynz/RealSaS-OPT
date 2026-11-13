package control

import (
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

func rssMeshParentFixture() ([]orchestration.ArtifactRef, []modelselection.Selection) {
	return []orchestration.ArtifactRef{{
		Role:          "input:0000",
		ArtifactType:  "RealSaS.RSSvNextTwoViewSource",
		SchemaVersion: "v1",
		ContentSHA256: strings.Repeat("c", 64),
	}}, []modelselection.Selection{{
		Family:                  "RSSVNEXT",
		ModelReleaseID:          uuid.MustParse("11111111-1111-4111-8111-111111111111"),
		ArchitectureID:          "RSS.vNext.UnifiedCharacterCompiler.v1",
		ModelEngineReleaseID:    uuid.MustParse("22222222-2222-4222-8222-222222222222"),
		SourceCodeSHA:           strings.Repeat("a", 40),
		CheckpointArtifactID:    uuid.MustParse("33333333-3333-4333-8333-333333333333"),
		CheckpointContentSHA256: strings.Repeat("b", 64),
		ArtifactType:            "RealSaS.RSSvNextUnifiedInferenceCheckpoint",
		ArtifactSchemaVersion:   "v1",
	}}
}

func TestRSSVNextMeshParentBindingReplaysExactPythonModelIdentity(t *testing.T) {
	refs, models := rssMeshParentFixture()
	got, err := rssVNextMeshParentBinding(refs, models)
	if err != nil {
		t.Fatal(err)
	}
	expected := "14aaceab6764e97d1ec26375f5e82033efda2e47f2f60dab704326a36a1caec3"
	if got["schema"] != "RealSaS.RSSvNext.ShapeParentBinding.v1" ||
		got["source_content_sha256"] != strings.Repeat("c", 64) ||
		got["checkpoint_sha256"] != strings.Repeat("b", 64) ||
		got["model_release_behavior_sha256"] != expected {
		t.Fatalf("RSS_VNEXT_GO_PYTHON_PARENT_SHA_DRIFT: %+v", got)
	}
}

func TestRSSVNextMeshParentBindingAcceptsSingleSealedReconstructionBundleV2(t *testing.T) {
	refs, models := rssMeshParentFixture()
	models[0].ArtifactSchemaVersion = "v2"
	result, err := rssVNextMeshParentBinding(refs, models)
	if err != nil {
		t.Fatal(err)
	}
	if result["checkpoint_sha256"] != strings.Repeat("b", 64) {
		t.Fatalf("checkpoint identity drift: %+v", result)
	}
	models[0].ArtifactSchemaVersion = "v3"
	if _, err := rssVNextMeshParentBinding(refs, models); err == nil ||
		!strings.Contains(err.Error(), "CHECKPOINT_ABI_DRIFT") {
		t.Fatalf("invalid checkpoint ABI admitted: %v", err)
	}
}

func TestRSSVNextMeshParentBindingRejectsWrongSourceCheckpointOrModelFamily(t *testing.T) {
	refs, models := rssMeshParentFixture()
	cases := []struct {
		name   string
		refs   []orchestration.ArtifactRef
		models []modelselection.Selection
		code   string
	}{
		{"no source", nil, models, "SOURCE_ARTIFACT_REQUIRED"},
		{"duplicate source", append(append([]orchestration.ArtifactRef{}, refs...), refs...), models, "SOURCE_ARTIFACT_CARDINALITY"},
		{"wrong source sha", []orchestration.ArtifactRef{{ArtifactType: "RealSaS.RSSvNextTwoViewSource", SchemaVersion: "v1", ContentSHA256: "bad"}}, models, "SOURCE_SHA_INVALID"},
		{"legacy families", refs, append(append([]modelselection.Selection{}, models...), modelselection.Selection{Family: "AXIS"}), "SINGLE_ATTEMPT_MODEL_REQUIRED"},
		{"wrong model checkpoint", refs, func() []modelselection.Selection {
			x := append([]modelselection.Selection{}, models...)
			x[0].CheckpointContentSHA256 = "broken"
			return x
		}(), "CHECKPOINT_SHA_INVALID"},
		{"wrong architecture", refs, func() []modelselection.Selection {
			x := append([]modelselection.Selection{}, models...)
			x[0].ArchitectureID = "legacy"
			return x
		}(), "UNIFIED_ARCHITECTURE_DRIFT"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := rssVNextMeshParentBinding(tc.refs, tc.models)
			if err == nil || !strings.Contains(err.Error(), tc.code) {
				t.Fatalf("expected %s, got %v", tc.code, err)
			}
		})
	}
}
