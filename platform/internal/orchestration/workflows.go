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
			InitialInterval: 2 * time.Second, BackoffCoefficient: 2,
			MaximumInterval: time.Minute, MaximumAttempts: 5,
		},
	})
	engineCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue:              EngineTaskQueue,
		StartToCloseTimeout:    4 * time.Hour,
		ScheduleToCloseTimeout: 12 * time.Hour,
		HeartbeatTimeout:       2 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval: 5 * time.Second, BackoffCoefficient: 2,
			MaximumInterval: 2 * time.Minute, MaximumAttempts: 3,
		},
	})

	var plan ResolvedCompilePlan
	if err := workflow.ExecuteActivity(platformCtx, ResolveCompilePlanActivityName, input).Get(ctx, &plan); err != nil {
		return CompileWorkflowResult{}, err
	}
	allowed := executeStageIDs(plan)
	completed := make([]StageCommitResult, 0, len(plan.Stages))
	signalChannel := workflow.GetSignalChannel(ctx, ExternalStageReadySignalName)

	refreshAt := func(index int, expectedStageID string) (ResolvedCompilePlan, error) {
		var refreshed ResolvedCompilePlan
		if err := workflow.ExecuteActivity(platformCtx, ResolveCompilePlanActivityName, input).Get(ctx, &refreshed); err != nil {
			return ResolvedCompilePlan{}, err
		}
		if refreshed.TargetStageID != plan.TargetStageID || len(refreshed.Stages) != len(plan.Stages) || index >= len(refreshed.Stages) || refreshed.Stages[index].StageID != expectedStageID {
			return ResolvedCompilePlan{}, temporal.NewNonRetryableApplicationError(
				"external evidence changed compile graph identity", "ExternalEvidencePlanDrift", nil,
			)
		}
		return refreshed, nil
	}

	for index := 0; index < len(plan.Stages); {
		stage := plan.Stages[index]
		switch stage.Action {
		case "REUSE":
			if stage.ReusableArtifactID == nil || *stage.ReusableArtifactID == "" {
				return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
					"reuse stage missing artifact identity", "ReuseContractViolation", nil,
				)
			}
			var committed StageCommitResult
			request := BindReusedStageRequest{
				AttemptID: input.AttemptID, StageID: stage.StageID,
				ExpectedSemanticSHA256: stage.ExpectedSemanticSHA256,
				ArtifactID:             *stage.ReusableArtifactID,
			}
			if err := workflow.ExecuteActivity(platformCtx, BindReusedStageActivityName, request).Get(ctx, &committed); err != nil {
				return CompileWorkflowResult{}, err
			}
			completed = append(completed, committed)
			index++

		case "EXECUTE":
			prepare := PrepareStageExecutionRequest{
				CommandID: input.CommandID, AttemptID: input.AttemptID,
				SubjectID: input.SubjectID, SubjectInputID: input.SubjectInputID,
				EngineReleaseID: input.EngineReleaseID, StageID: stage.StageID,
				ExpectedSemanticSHA256: stage.ExpectedSemanticSHA256,
				AllowedExecuteStageIDs: allowed,
			}
			var request EngineStageRequest
			if err := workflow.ExecuteActivity(platformCtx, PrepareStageExecutionActivityName, prepare).Get(ctx, &request); err != nil {
				return CompileWorkflowResult{}, err
			}
			var engineResult EngineStageResult
			if err := workflow.ExecuteActivity(engineCtx, EngineExecuteStageActivityName, request).Get(ctx, &engineResult); err != nil {
				_ = workflow.ExecuteActivity(platformCtx, RecordStageActivityErrorActivityName, StageActivityErrorRequest{
					ExecutionID: request.ExecutionID, AttemptID: input.AttemptID,
					StageID: stage.StageID, Error: err.Error(),
				}).Get(ctx, nil)
				return CompileWorkflowResult{}, err
			}
			var committed StageCommitResult
			if err := workflow.ExecuteActivity(platformCtx, CommitStageResultActivityName, StageCommitRequest{
				ExecutionID: request.ExecutionID, CommandID: input.CommandID,
				AttemptID: input.AttemptID, StageID: stage.StageID,
				ExpectedSemanticSHA256: stage.ExpectedSemanticSHA256,
				AllowedExecuteStageIDs: allowed, EngineResult: engineResult,
			}).Get(ctx, &committed); err != nil {
				return CompileWorkflowResult{}, err
			}

			if engineResult.Status == "WAITING_EXTERNAL_EVIDENCE" {
				// Re-resolve before blocking. Evidence can arrive after the Engine
				// observed it missing but before the WAITING DB status commits; in
				// that race the binder intentionally sends no signal. Semantic SHA
				// change is the durable lost-wakeup guard.
				refreshed, err := refreshAt(index, stage.StageID)
				if err != nil {
					return CompileWorkflowResult{}, err
				}
				if refreshed.Stages[index].ExpectedSemanticSHA256 == stage.ExpectedSemanticSHA256 {
					var signal ExternalStageReadySignal
					signalChannel.Receive(ctx, &signal)
					if signal.AttemptID != input.AttemptID || signal.StageID != stage.StageID || signal.EvidenceSHA256 == "" {
						return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
							"external evidence signal identity drift", "ExternalEvidenceSignalViolation", nil,
						)
					}
					refreshed, err = refreshAt(index, stage.StageID)
					if err != nil {
						return CompileWorkflowResult{}, err
					}
				}
				plan = refreshed
				allowed = executeStageIDs(plan)
				continue
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
			index++

		default:
			return CompileWorkflowResult{}, temporal.NewNonRetryableApplicationError(
				"unknown compile resolution action: "+stage.Action,
				"CompilePlanContractViolation", nil,
			)
		}
	}

	var result CompileWorkflowResult
	if err := workflow.ExecuteActivity(platformCtx, FinalizeCompileActivityName, map[string]any{
		"command": input, "plan": plan, "completed_stages": completed,
	}).Get(ctx, &result); err != nil {
		return CompileWorkflowResult{}, err
	}
	return result, nil
}

func RenderWorkflow(ctx workflow.Context, input RenderWorkflowInput) (RenderWorkflowResult, error) {
	platformCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue: ControlTaskQueue, StartToCloseTimeout: 5 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{InitialInterval: 2 * time.Second, BackoffCoefficient: 2, MaximumInterval: time.Minute, MaximumAttempts: 5},
	})
	engineCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		TaskQueue: EngineTaskQueue, StartToCloseTimeout: 2 * time.Hour,
		ScheduleToCloseTimeout: 4 * time.Hour, HeartbeatTimeout: time.Minute,
		RetryPolicy: &temporal.RetryPolicy{InitialInterval: 5 * time.Second, BackoffCoefficient: 2, MaximumInterval: time.Minute, MaximumAttempts: 3},
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
		return RenderWorkflowResult{}, temporal.NewNonRetryableApplicationError("runtime render failed", "RenderRuntimeFailure", nil)
	}
	var result RenderWorkflowResult
	if err := workflow.ExecuteActivity(platformCtx, CommitRenderOutputActivityName, RenderCommitRequest{
		CommandID: input.CommandID, RenderRequestID: input.RenderRequestID, EngineResult: engineResult,
	}).Get(ctx, &result); err != nil {
		return RenderWorkflowResult{}, err
	}
	return result, nil
}

func RenderActivityNames() []string {
	return []string{ResolveRenderRequestActivityName, EngineRenderTailActivityName, CommitRenderOutputActivityName}
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
