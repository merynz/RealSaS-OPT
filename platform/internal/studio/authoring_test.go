package studio

import (
	"encoding/json"
	"errors"
	"testing"
)

func validLayer() json.RawMessage {
	return json.RawMessage(`{
		"schema_version":"RealSaS.UserPuppetEditLayer.v3",
		"source_product_state_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"authority_class":"OPTIONAL_USER_AUTHORING_LAYER",
		"can_promote_product_directly":false,
		"requires_compiler_requalification":true,
		"metadata":{"canonical_bundle_mutated":false},
		"rig_pose_edits":[],
		"mesh_vertex_edits":[],
		"weight_edits":[],
		"binding_edits":[],
		"rig_topology_edits":[],
		"ik_constraints":[],
		"animation_edits":[]
	}`)
}

func TestCanonicalAuthoringPayloadAcceptsSiblingLayer(t *testing.T) {
	payload, digest, err := canonicalAuthoringPayload(validLayer())
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) == 0 || len(digest) != 64 {
		t.Fatalf("unexpected canonical payload/hash: %d %q", len(payload), digest)
	}
	_, digest2, err := canonicalAuthoringPayload(json.RawMessage(`{
		"animation_edits":[],"ik_constraints":[],"rig_topology_edits":[],"binding_edits":[],
		"weight_edits":[],"mesh_vertex_edits":[],"rig_pose_edits":[],
		"metadata":{"canonical_bundle_mutated":false},
		"requires_compiler_requalification":true,"can_promote_product_directly":false,
		"authority_class":"OPTIONAL_USER_AUTHORING_LAYER",
		"source_product_state_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"schema_version":"RealSaS.UserPuppetEditLayer.v3"
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if digest != digest2 {
		t.Fatalf("canonical hash must ignore object key order: %s != %s", digest, digest2)
	}
}

func TestCanonicalAuthoringPayloadRejectsAuthorityEscalation(t *testing.T) {
	var value map[string]any
	if err := json.Unmarshal(validLayer(), &value); err != nil {
		t.Fatal(err)
	}
	value["can_promote_product_directly"] = true
	raw, _ := json.Marshal(value)
	if _, _, err := canonicalAuthoringPayload(raw); !errors.Is(err, ErrAuthoringPayloadInvalid) {
		t.Fatalf("expected authority rejection, got %v", err)
	}
}

func TestCanonicalAuthoringPayloadRejectsCanonicalMutation(t *testing.T) {
	var value map[string]any
	if err := json.Unmarshal(validLayer(), &value); err != nil {
		t.Fatal(err)
	}
	value["metadata"].(map[string]any)["canonical_bundle_mutated"] = true
	raw, _ := json.Marshal(value)
	if _, _, err := canonicalAuthoringPayload(raw); !errors.Is(err, ErrAuthoringPayloadInvalid) {
		t.Fatalf("expected canonical mutation rejection, got %v", err)
	}
}

func TestPublicProjectIDsAreOpaque(t *testing.T) {
	id := newProjectID()
	if !projectIDPattern.MatchString(id) {
		t.Fatalf("invalid public project id %q", id)
	}
}
