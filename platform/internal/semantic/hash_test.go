package semantic

import "testing"

func TestArtifactCanonicalHashMatchesPythonReference(t *testing.T) {
	payload := map[string]any{
		"artifact_type":         "RealSaS.QualifiedMeshIR",
		"schema_version":        "v1",
		"producer_contract":     "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
		"implementation_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
		"policy_sha256":         "2222222222222222222222222222222222222222222222222222222222222222",
		"inputs": []any{
			map[string]any{
				"role":            "skeleton",
				"ordinal":         0,
				"artifact_type":   "RealSaS.QualifiedSkeletonIR",
				"semantic_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
			},
		},
		"semantic_parameters": map[string]any{"b": 2, "a": 1},
	}
	got, err := JSONSHA256(payload)
	if err != nil {
		t.Fatal(err)
	}
	const want = "9dce46bdf5ae8c0adacd24d50447adb15806e00def3578e14906279100eff43c"
	if got != want {
		t.Fatalf("Go/Python semantic hash drift: got %s want %s", got, want)
	}
}
