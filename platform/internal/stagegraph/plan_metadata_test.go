package stagegraph

import (
	"encoding/json"
	"testing"
)

func TestReleasedGraphSnapshotPreservesPlanMetadata(t *testing.T) {
	raw := []byte(`{
	  "schema":"RealSaS.StageGraph.v1",
	  "plan_metadata":{
	    "schema":"RealSaS.MainlineExecutionPlan.v2",
	    "canonical_branch":"main",
	    "architecture_scope":"RESEARCH_STAGE48"
	  },
	  "stage_count":1,
	  "stages":[{
	    "ordinal":1,
	    "id":"01_SOURCE_BYTES_SEALED",
	    "title":"Source bytes sealed",
	    "group":"source",
	    "depends_on":[],
	    "adapter":"x:y",
	    "manifest_keys":["source_files"],
	    "policy":{"fail_closed":true,"cacheable":true,"output_hash_required":true,"product_pass_authority":false}
	  }]
	}`)
	g, err := ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := g.Snapshot()
	if got := snapshot.PlanMetadata["architecture_scope"]; got != "RESEARCH_STAGE48" {
		t.Fatalf("plan metadata lost: %v", got)
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip Snapshot
	if err := json.Unmarshal(encoded, &roundtrip); err != nil {
		t.Fatal(err)
	}
	if got := roundtrip.PlanMetadata["canonical_branch"]; got != "main" {
		t.Fatalf("roundtrip metadata drift: %v", got)
	}
}
