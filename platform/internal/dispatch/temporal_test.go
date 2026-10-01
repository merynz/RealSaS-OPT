package dispatch

import (
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

func TestCompileWorkflowIdentityIsDeterministicFromCommandID(t *testing.T) {
	payload := map[string]any{
		"command_id":        "cmd-123",
		"attempt_id":        "attempt-1",
		"subject_id":        "subject-1",
		"engine_release_id": "release-1",
		"subject_input_id":  "input-1",
		"target_stage_id":   "46_PRODUCT_CLOSURE_SEAL",
	}
	name, id, _, input, err := workflowSpec(command.CompileSubject, payload)
	if err != nil {
		t.Fatal(err)
	}
	if name != orchestration.CompileWorkflowName || id != "realsas:compile:cmd-123" {
		t.Fatalf("name=%s id=%s", name, id)
	}
	got := input.(orchestration.CompileWorkflowInput)
	if got.AttemptID != "attempt-1" || got.TargetStageID != "46_PRODUCT_CLOSURE_SEAL" {
		t.Fatalf("input=%+v", got)
	}
}

func TestRenderWorkflowIdentityIsDeterministicFromCommandID(t *testing.T) {
	payload := map[string]any{
		"command_id":                     "cmd-456",
		"subject_id":                     "subject-1",
		"product_revision_id":            "revision-7",
		"render_request_id":              "render-1",
		"render_request_semantic_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	name, id, _, input, err := workflowSpec(command.RenderProduct, payload)
	if err != nil {
		t.Fatal(err)
	}
	if name != orchestration.RenderWorkflowName || id != "realsas:render:cmd-456" {
		t.Fatalf("name=%s id=%s", name, id)
	}
	got := input.(orchestration.RenderWorkflowInput)
	if got.ProductRevisionID != "revision-7" || got.RenderRequestID != "render-1" {
		t.Fatalf("input=%+v", got)
	}
}
