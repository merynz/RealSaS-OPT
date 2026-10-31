package orchestration

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
)

func TestProductCapabilityWorkflowRunsProofTerminalAndReturnsProductRevision(t *testing.T) {
	var suite testsuite.WorkflowTestSuite
	env := suite.NewTestWorkflowEnvironment()
	defer env.Close()

	input := CapabilityWorkflowInput{
		CommandID:       "11111111-1111-4111-8111-111111111111",
		AttemptID:       "22222222-2222-4222-8222-222222222222",
		ExecutionGoalID: "33333333-3333-4333-8333-333333333333",
		EngineReleaseID: "44444444-4444-4444-8444-444444444444",
		GoalSpecSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	capabilityIDs := []string{
		"product.input",
		"product.shape",
		"product.mesh",
		"product.rig",
		"product.skin",
		"product.mechanics",
		"product.graphics",
		"product.motion",
		"product.proof",
	}
	steps := make([]CapabilityStep, 0, len(capabilityIDs))
	for index, capabilityID := range capabilityIDs {
		dependencies := []string(nil)
		if index > 0 {
			dependencies = []string{capabilityIDs[index-1]}
		}
		steps = append(steps, CapabilityStep{
			CapabilityID:         capabilityID,
			Kind:                 "MODULE",
			OwnerModuleID:        "engine." + capabilityID,
			ExecutorActivity:     EngineExecuteCapabilityActivityName,
			ImplementationSHA256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			PolicySHA256:         "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
			ParametersSHA256:     "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
			Dependencies:         dependencies,
		})
	}
	goal := ResolvedCapabilityGoal{
		ExecutionGoalID:     input.ExecutionGoalID,
		GoalType:            "PRODUCT_COMPILE",
		PromotionPolicy:     "QUALIFIED_PRODUCT_ONLY",
		CapabilitySetSHA256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
		Parameters:          map[string]any{"source": "court"},
		Steps:               steps,
	}

	var prepared []string
	var executed []string
	var committed []string
	productFinalizerCalls := 0
	legacyFinalizerCalls := 0

	env.RegisterActivityWithOptions(
		func(context.Context, CapabilityWorkflowInput) (ResolvedCapabilityGoal, error) {
			return goal, nil
		},
		activity.RegisterOptions{Name: ResolveCapabilityGoalActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req PrepareCapabilityExecutionRequest) (EngineCapabilityRequest, error) {
			prepared = append(prepared, req.Step.CapabilityID)
			return EngineCapabilityRequest{
				ExecutionID:               "execution-" + req.Step.CapabilityID,
				CommandID:                 req.CommandID,
				AttemptID:                 req.AttemptID,
				ExecutionGoalID:           req.ExecutionGoalID,
				EngineReleaseID:           req.EngineReleaseID,
				CapabilityID:              req.Step.CapabilityID,
				Kind:                      req.Step.Kind,
				OwnerModuleID:             req.Step.OwnerModuleID,
				RequestedExecutorActivity: req.Step.ExecutorActivity,
				ImplementationSHA256:      req.Step.ImplementationSHA256,
				PolicySHA256:              req.Step.PolicySHA256,
				ParametersSHA256:          req.Step.ParametersSHA256,
			}, nil
		},
		activity.RegisterOptions{Name: PrepareCapabilityExecutionActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req EngineCapabilityRequest) (EngineCapabilityResult, error) {
			executed = append(executed, req.CapabilityID)
			return EngineCapabilityResult{CapabilityID: req.CapabilityID, Status: "PASS"}, nil
		},
		activity.RegisterOptions{Name: EngineExecuteCapabilityActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req CapabilityCommitRequest) (CapabilityCommitResult, error) {
			committed = append(committed, req.Request.CapabilityID)
			if req.Result.CapabilityID != req.Request.CapabilityID || req.Result.Status != "PASS" {
				return CapabilityCommitResult{}, errors.New("court received invalid engine result")
			}
			return CapabilityCommitResult{CapabilityID: req.Request.CapabilityID, Status: "PASS"}, nil
		},
		activity.RegisterOptions{Name: CommitCapabilityResultActivityName},
	)
	env.RegisterActivityWithOptions(
		func(context.Context, map[string]any) (CapabilityWorkflowResult, error) {
			legacyFinalizerCalls++
			return CapabilityWorkflowResult{}, errors.New("legacy capability finalizer must not own product promotion")
		},
		activity.RegisterOptions{Name: FinalizeCapabilityGoalActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, request map[string]any) (CapabilityWorkflowResult, error) {
			productFinalizerCalls++
			if request["command"] == nil || request["goal"] == nil || request["completed"] == nil {
				return CapabilityWorkflowResult{}, errors.New("product finalizer request incomplete")
			}
			revisionID := "55555555-5555-4555-8555-555555555555"
			completed := make([]CapabilityCommitResult, 0, len(capabilityIDs))
			for _, capabilityID := range capabilityIDs {
				completed = append(completed, CapabilityCommitResult{CapabilityID: capabilityID, Status: "PASS"})
			}
			return CapabilityWorkflowResult{
				AttemptID:         input.AttemptID,
				Status:            "PASS",
				Completed:         completed,
				ProductRevisionID: &revisionID,
			}, nil
		},
		activity.RegisterOptions{Name: FinalizeCapabilityProductActivityName},
	)

	env.ExecuteWorkflow(CapabilityWorkflow, input)
	if err := env.GetWorkflowError(); err != nil {
		t.Fatal(err)
	}
	var result CapabilityWorkflowResult
	if err := env.GetWorkflowResult(&result); err != nil {
		t.Fatal(err)
	}
	if result.Status != "PASS" || result.ProductRevisionID == nil || *result.ProductRevisionID == "" {
		t.Fatalf("product workflow result=%+v", result)
	}
	if productFinalizerCalls != 1 || legacyFinalizerCalls != 0 {
		t.Fatalf("finalizers product=%d legacy=%d", productFinalizerCalls, legacyFinalizerCalls)
	}
	if !reflect.DeepEqual(prepared, capabilityIDs) {
		t.Fatalf("prepared=%v want=%v", prepared, capabilityIDs)
	}
	if !reflect.DeepEqual(executed, capabilityIDs) {
		t.Fatalf("executed=%v want=%v", executed, capabilityIDs)
	}
	if !reflect.DeepEqual(committed, capabilityIDs) {
		t.Fatalf("committed=%v want=%v", committed, capabilityIDs)
	}
}

func TestProductCapabilityWorkflowDoesNotFinalizeWhenProofFails(t *testing.T) {
	var suite testsuite.WorkflowTestSuite
	env := suite.NewTestWorkflowEnvironment()
	defer env.Close()

	input := CapabilityWorkflowInput{
		CommandID:       "11111111-1111-4111-8111-111111111111",
		AttemptID:       "22222222-2222-4222-8222-222222222222",
		ExecutionGoalID: "33333333-3333-4333-8333-333333333333",
		EngineReleaseID: "44444444-4444-4444-8444-444444444444",
		GoalSpecSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	goal := ResolvedCapabilityGoal{
		ExecutionGoalID: input.ExecutionGoalID,
		GoalType:        "PRODUCT_COMPILE",
		PromotionPolicy: "QUALIFIED_PRODUCT_ONLY",
		Steps: []CapabilityStep{
			{
				CapabilityID:         "product.proof",
				Kind:                 "MODULE",
				OwnerModuleID:        "engine.product.proof",
				ExecutorActivity:     EngineExecuteCapabilityActivityName,
				ImplementationSHA256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
				PolicySHA256:         "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
				ParametersSHA256:     "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
			},
		},
	}
	finalizerCalls := 0

	env.RegisterActivityWithOptions(
		func(context.Context, CapabilityWorkflowInput) (ResolvedCapabilityGoal, error) {
			return goal, nil
		},
		activity.RegisterOptions{Name: ResolveCapabilityGoalActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req PrepareCapabilityExecutionRequest) (EngineCapabilityRequest, error) {
			return EngineCapabilityRequest{
				ExecutionID:               "execution-product.proof",
				CommandID:                 req.CommandID,
				AttemptID:                 req.AttemptID,
				ExecutionGoalID:           req.ExecutionGoalID,
				EngineReleaseID:           req.EngineReleaseID,
				CapabilityID:              req.Step.CapabilityID,
				Kind:                      req.Step.Kind,
				OwnerModuleID:             req.Step.OwnerModuleID,
				RequestedExecutorActivity: req.Step.ExecutorActivity,
				ImplementationSHA256:      req.Step.ImplementationSHA256,
				PolicySHA256:              req.Step.PolicySHA256,
				ParametersSHA256:          req.Step.ParametersSHA256,
			}, nil
		},
		activity.RegisterOptions{Name: PrepareCapabilityExecutionActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req EngineCapabilityRequest) (EngineCapabilityResult, error) {
			return EngineCapabilityResult{
				CapabilityID: req.CapabilityID,
				Status:       "FAIL",
				Failure: &EngineCapabilityFailure{
					Code:  "PRODUCT_PROOF_FAILED",
					Class: "PROOF",
				},
			}, nil
		},
		activity.RegisterOptions{Name: EngineExecuteCapabilityActivityName},
	)
	env.RegisterActivityWithOptions(
		func(_ context.Context, req CapabilityCommitRequest) (CapabilityCommitResult, error) {
			return CapabilityCommitResult{CapabilityID: req.Request.CapabilityID, Status: "FAIL"}, nil
		},
		activity.RegisterOptions{Name: CommitCapabilityResultActivityName},
	)
	env.RegisterActivityWithOptions(
		func(context.Context, map[string]any) (CapabilityWorkflowResult, error) {
			finalizerCalls++
			return CapabilityWorkflowResult{}, nil
		},
		activity.RegisterOptions{Name: FinalizeCapabilityProductActivityName},
	)

	env.ExecuteWorkflow(CapabilityWorkflow, input)
	if err := env.GetWorkflowError(); err == nil {
		t.Fatal("proof failure must fail product capability workflow")
	}
	if finalizerCalls != 0 {
		t.Fatalf("product finalizer called after proof failure: %d", finalizerCalls)
	}
}
