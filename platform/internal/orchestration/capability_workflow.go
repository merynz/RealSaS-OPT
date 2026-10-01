package orchestration

import (
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"
)

func CapabilityWorkflow(ctx workflow.Context, input CapabilityWorkflowInput) (CapabilityWorkflowResult, error) {
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

	var goal ResolvedCapabilityGoal
	if err := workflow.ExecuteActivity(platformCtx, ResolveCapabilityGoalActivityName, input).Get(ctx, &goal); err != nil {
		return CapabilityWorkflowResult{}, err
	}
	if goal.PromotionPolicy != "NEVER" {
		return CapabilityWorkflowResult{}, temporal.NewNonRetryableApplicationError(
			"developer capability workflow cannot promote product state",
			"CapabilityPromotionContractViolation",
			nil,
		)
	}

	completed := make([]CapabilityCommitResult, 0, len(goal.Steps))
	for _, step := range goal.Steps {
		var request EngineCapabilityRequest
		if err := workflow.ExecuteActivity(platformCtx, PrepareCapabilityExecutionActivityName, PrepareCapabilityExecutionRequest{
			CommandID:       input.CommandID,
			AttemptID:       input.AttemptID,
			ExecutionGoalID: input.ExecutionGoalID,
			EngineReleaseID: input.EngineReleaseID,
			Step:            step,
			GoalParameters:  goal.Parameters,
		}).Get(ctx, &request); err != nil {
			return CapabilityWorkflowResult{}, err
		}

		var engineResult EngineCapabilityResult
		if err := workflow.ExecuteActivity(engineCtx, EngineExecuteCapabilityActivityName, request).Get(ctx, &engineResult); err != nil {
			_ = workflow.ExecuteActivity(platformCtx, RecordCapabilityActivityErrorActivityName, CapabilityActivityErrorRequest{
				ExecutionID:   request.ExecutionID,
				AttemptID:     input.AttemptID,
				CapabilityID:  step.CapabilityID,
				OwnerModuleID: step.OwnerModuleID,
				Error:         err.Error(),
			}).Get(ctx, nil)
			return CapabilityWorkflowResult{}, err
		}

		var committed CapabilityCommitResult
		if err := workflow.ExecuteActivity(platformCtx, CommitCapabilityResultActivityName, CapabilityCommitRequest{
			Request: request,
			Result:  engineResult,
		}).Get(ctx, &committed); err != nil {
			return CapabilityWorkflowResult{}, err
		}
		completed = append(completed, committed)
		if engineResult.Status != "PASS" {
			code := "CAPABILITY_FAILED"
			if engineResult.Failure != nil && engineResult.Failure.Code != "" {
				code = engineResult.Failure.Code
			}
			return CapabilityWorkflowResult{}, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("%s:%s", step.CapabilityID, code),
				"CapabilityExecutionFailure",
				nil,
			)
		}
	}

	var result CapabilityWorkflowResult
	if err := workflow.ExecuteActivity(platformCtx, FinalizeCapabilityGoalActivityName, map[string]any{
		"command":   input,
		"goal":      goal,
		"completed": completed,
	}).Get(ctx, &result); err != nil {
		return CapabilityWorkflowResult{}, err
	}
	return result, nil
}
