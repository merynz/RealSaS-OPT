package control

import (
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

func TestProductRevisionModelSnapshotChangesManifestIdentity(t *testing.T) {
	bindings := []revisionBinding{{
		Role:           "runtime:package",
		ArtifactID:     "11111111-1111-1111-1111-111111111111",
		ArtifactType:   "RealSaS.RuntimePackageArtifact",
		SchemaVersion:  "v1",
		SemanticSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}}
	base := productRevisionManifest(
		"22222222-2222-2222-2222-222222222222",
		"33333333-3333-3333-3333-333333333333",
		"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		bindings,
		nil,
	)
	if base["schema"] != "RealSaS.ProductRevisionManifest.v1" {
		t.Fatalf("legacy manifest schema drift: %v", base["schema"])
	}
	if _, ok := base["model_releases"]; ok {
		t.Fatal("legacy ProductRevision identity unexpectedly gained model rows")
	}

	modelA := []map[string]any{{
		"family":                    "iris",
		"model_release_id":          "44444444-4444-4444-4444-444444444444",
		"checkpoint_content_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
	}}
	modelB := []map[string]any{{
		"family":                    "iris",
		"model_release_id":          "55555555-5555-5555-5555-555555555555",
		"checkpoint_content_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
	}}
	manifestA := productRevisionManifest(
		base["subject_id"].(string), base["engine_release_id"].(string), base["product_contract_sha256"].(string), bindings, modelA,
	)
	manifestB := productRevisionManifest(
		base["subject_id"].(string), base["engine_release_id"].(string), base["product_contract_sha256"].(string), bindings, modelB,
	)
	if manifestA["schema"] != "RealSaS.ProductRevisionManifest.v2" || manifestB["schema"] != "RealSaS.ProductRevisionManifest.v2" {
		t.Fatal("model-bound revision did not use v2 identity schema")
	}
	shaA, err := semantic.JSONSHA256(manifestA)
	if err != nil {
		t.Fatal(err)
	}
	shaB, err := semantic.JSONSHA256(manifestB)
	if err != nil {
		t.Fatal(err)
	}
	if shaA == shaB {
		t.Fatal("different exact model snapshots aliased the same ProductRevision identity")
	}
}
