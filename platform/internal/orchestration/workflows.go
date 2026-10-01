package orchestration

import (
	"fmt"
	"strings"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"
)

func executeStageIDs(plan ResolvedCompilePlan) []string {
	out := make([]string, 0)
	for _, stage := range plan.Stages {
		if stage.Action == "EXECUTE" {
			out = append(out, stage.StageID)
		}
	}
	return out
}

func CompileWorkflow(ctx workflow.Context, input CompileWorkflowInput) (CompileWorkflowResult, error) {
	platformCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue:              ControlTaskQueue,
		StartToCloseTimeout:    5 * time.Minute,
		ScheduleToCloseTimeout: 15 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    2 * time.Second,
			BackoffCoefficient: 2,
			MaximumInterval:    time.Minute,
			MaximumAttempts:    5,
		},
	})
	engineCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue:              EngineTaskQueue,
		StartToCloseTimeout:    4 * time.Hour,
		ScheduleToCloseTimeout: 12 * time.Hour,
		HeartbeatTimeout:       2 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    5 * time.Second,
			BackoffCoefficient: 2,
			MaximumInterval:    2 * time.Minute,
			MaximumAttempts:    3,
		},
	})

	var plan ResolvedCompilePlan
	if err := workflow.ExecuteActivity(platformCtx, ResolveCompilePlanActivityName, input).Get(ctx, &plan); err != nil {
		return CompileWorkflowResult{}, err
	}
	allowed := executeStageIDs(plan)
	completed := make([]StageCommitResult, 0, len(plan.Stages))

	for _, stage := range plan.Stages {
		switch stage.Action {
		case "REUSE":
			if stage.ReusableArtifactID == nil || *stage.ReusableArtifactID == "" {
				return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
					"reuse stage missing artifact identity", "ReuseContractViolation", nil,
				)
			}
			var committed StageCommitResult
			request := map[string]any{
				"attempt_id":               input.AttemptID,
				"stage_id":                 stage.StageID,
				"expected_semantic_sha256": stage.ExpectedSemanticSHA256,
				"artifact_id":              *stage.ReusableArtifactID,
			}
			if err := workflow.ExecuteActivity(platformCtx, BindReusedStageActivityName, request).Get(ctx, &committed); err != nil {
				return CompileWorkflowResult{}, err
			}
			completed = append(completed, committed)

		case "EXECUTE":
			request := EngineStageRequest{
				CommandID:              input.CommandID,
				AttemptID:              input.AttemptID,
				SubjectID:              input.SubjectID,
				EngineReleaseID:        input.EngineReleaseID,
				StageID:                stage.StageID,
				ExpectedSemanticSHA256: stage.ExpectedSemanticSHA256,
				AllowedExecuteStageIDs: allowed,
			}
			var engineResult EngineStageResult
			if err := workflow.ExecuteActivity(engineCtx, EngineExecuteStageActivityName, request).Get(ctx, &engineResult); err != nil {
				return CompileWorkflowResult{}, err
			}
			var committed StageCommitResult
			if err := workflow.ExecuteActivity(platformCtx, CommitStageResultActivityName, StageCommitRequest{
				CommandID:              input.CommandID,
				AttemptID:              input.AttemptID,
				StageID:                stage.StageID,
				ExpectedSemanticSHA256: stage.ExpectedSemanticSHA256,
				EngineResult:           engineResult,
			}).Get(ctx, &committed); err != nil {
				return CompileWorkflowResult{}, err
			}
			completed = append(completed, committed)
			if engineResult.Status != "PASS" && engineResult.Status != "PASS_DEMO_ONLY" {
				code := "COMPILER_STAGE_FAILED"
				if engineResult.Failure != nil && engineResult.Failure.Code != "" {
					code = engineResult.Failure.Code
				}
				return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
					fmt.Sprintf("%s:%s", stage.StageID, code), "CompilerStageFailure", nil,
				)
			}
		default:
			return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
				"unknown compile resolution action: "+stage.Action,
				"CompilePlanContractViolation",
				nil,
			)
		}
	}

	var result CompileWorkflowResult
	if err := workflow.ExecuteActivity(platformCtx, FinalizeCompileActivityName, map[string]any{
		"command":          input,
		"plan":             plan,
		"completed_stages": completed,
	}).Get(ctx, &result); err != nil {
		return CompileWorkflowResult{}, err
	}
	return result, nil
}

func RenderWorkflow(ctx workflow.Context, input RenderWorkflowInput) (RenderWorkflowResult, error) {
	platformCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue:           ControlTaskQueue,
		StartToCloseTimeout: 5 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    2 * time.Second,
			BackoffCoefficient: 2,
			MaximumInterval:    time.Minute,
			MaximumAttempts:    5,
		},
	})
	engineCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue:              EngineTaskQueue,
		StartToCloseTimeout:    2 * time.Hour,
		ScheduleToCloseTimeout: 4 * time.Hour,
		HeartbeatTimeout:       time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    5 * time.Second,
			BackoffCoefficient: 2,
			MaximumInterval:    time.Minute,
			MaximumAttempts:    3,
		},
	})

	var resolution RenderResolution
	if err := workflow.ExecuteActivity(platformCtx, ResolveRenderRequestActivityName, input).Get(ctx, &resolution); err != nil {
		return RenderWorkflowResult{}, err
	}
	if resolution.CacheHit {
		if resolution.OutputArtifactID == nil || *resolution.OutputArtifactID == "" {
			return RenderWorkflowResult{}, temporal.NewNonRetryableApplicationError(
				"render cache hit missing output artifact", "RenderContractViolation", nil,
			)
		}
		return RenderWorkflowResult{Status: "PASS", CacheHit: true, OutputArtifactID: *resolution.OutputArtifactID}, nil
	}

	var engineResult EngineRenderResult
	if err := workflow.ExecuteActivity(engineCtx, EngineRenderTailActivityName, resolution.RuntimeRequest).Get(ctx, &engineResult); err != nil {
		return RenderWorkflowResult{}, err
	}
	if engineResult.Status != "PASS" {
		return RenderWorkflowResult{}, temporal.NewNonRetryableApplicationError(
			"runtime render failed", "RenderRuntimeFailure", nil,
		)
	}
	var result RenderWorkflowResult
	if err := workflow.ExecuteActivity(platformCtx, CommitRenderOutputActivityName, RenderCommitRequest{
		CommandID:       input.CommandID,
		RenderRequestID: input.RenderRequestID,
		EngineResult:    engineResult,
	}).Get(ctx, &result); err != nil {
		return RenderWorkflowResult{}, err
	}
	return result, nil
}

func RenderActivityNames() []string {
	return []string{
		ResolveRenderRequestActivityName,
		EngineRenderTailActivityName,
		CommitRenderOutputActivityName,
	}
}

func RenderHasForbiddenCapability() bool {
	for _, name := range RenderActivityNames() {
		lower := strings.ToLower(name)
		for _, forbidden := range []string{"fit", "train", "calibrat", "promot", "compile_stage"} {
			if strings.Contains(lower, forbidden) {
				return true
			}
		}
	}
	return false
}
