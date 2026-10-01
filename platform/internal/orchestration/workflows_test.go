package orchestration

import "testing"

func TestRenderWorkflowHasNoTrainingOrCompileCapability(t *testing.T) {
	if RenderHasForbiddenCapability() {
		t.Fatalf("render activity surface contains forbidden capability: %v", RenderActivityNames())
	}
	want := []string{
		"platform.resolve_render_request.v1",
		"engine.render_runtime_tail.v1",
		"platform.commit_render_output.v1",
	}
	got := RenderActivityNames()
	if len(got) != len(want) {
		t.Fatalf("got=%v want=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
}

func TestCompilePlanRoutesReuseAndExecuteSeparately(t *testing.T) {
	plan := ResolvedCompilePlan{Stages: []ResolvedStage{
		{StageID: "34_DEFORMATION_CAPABILITY_ENVELOPE", Action: "REUSE"},
		{StageID: "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED", Action: "EXECUTE"},
		{StageID: "36_QUALIFIED_MESH_SKIN_TRANSFER", Action: "EXECUTE"},
	}}
	got := executeStageIDs(plan)
	want := []string{"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED", "36_QUALIFIED_MESH_SKIN_TRANSFER"}
	if len(got) != len(want) {
		t.Fatalf("got=%v want=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
}
