package dispatch

import (
	"context"
	"errors"
	"fmt"
	"time"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/api/serviceerror"
	"go.temporal.io/sdk/client"

	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

type TemporalPublisher struct {
	Client client.Client
}

func (p TemporalPublisher) Publish(ctx context.Context, commandType string, payload map[string]any) error {
	if p.Client == nil {
		return errors.New("temporal client is required")
	}
	workflowName, workflowID, timeout, input, err := workflowSpec(commandType, payload)
	if err != nil {
		return err
	}
	_, err = p.Client.ExecuteWorkflow(ctx, client.StartWorkflowOptions{
		ID:                       workflowID,
		TaskQueue:                orchestration.ControlTaskQueue,
		WorkflowExecutionTimeout: timeout,
		WorkflowIDReusePolicy:    enumspb.WORKFLOW_ID_REUSE_POLICY_REJECT_DUPLICATE,
	}, workflowName, input)
	if err == nil {
		return nil
	}
	var already *serviceerror.WorkflowExecutionAlreadyStarted
	if errors.As(err, &already) {
		return nil
	}
	return err
}

func workflowSpec(commandType string, payload map[string]any) (string, string, time.Duration, any, error) {
	commandID, err := requiredString(payload, "command_id")
	if err != nil {
		return "", "", 0, nil, err
	}
	switch commandType {
	case command.CompileSubject:
		attemptID, err := requiredString(payload, "attempt_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		subjectID, err := requiredString(payload, "subject_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		releaseID, err := requiredString(payload, "engine_release_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		inputID, err := requiredString(payload, "subject_input_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		target, err := requiredString(payload, "target_stage_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		return orchestration.CompileWorkflowName,
			"realsas:compile:" + commandID,
			7 * 24 * time.Hour,
			orchestration.CompileWorkflowInput{
				CommandID:       commandID,
				AttemptID:       attemptID,
				SubjectID:       subjectID,
				EngineReleaseID: releaseID,
				SubjectInputID:  inputID,
				TargetStageID:   target,
			}, nil
	case command.RunCapability:
		attemptID, err := requiredString(payload, "attempt_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		goalID, err := requiredString(payload, "execution_goal_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		releaseID, err := requiredString(payload, "engine_release_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		specSHA, err := requiredString(payload, "spec_sha256")
		if err != nil {
			return "", "", 0, nil, err
		}
		return orchestration.CapabilityWorkflowName,
			"realsas:capability:" + commandID,
			7 * 24 * time.Hour,
			orchestration.CapabilityWorkflowInput{
				CommandID:       commandID,
				AttemptID:       attemptID,
				ExecutionGoalID: goalID,
				EngineReleaseID: releaseID,
				GoalSpecSHA256:  specSHA,
			}, nil
	case command.RenderProduct:
		subjectID, err := requiredString(payload, "subject_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		revisionID, err := requiredString(payload, "product_revision_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		renderID, err := requiredString(payload, "render_request_id")
		if err != nil {
			return "", "", 0, nil, err
		}
		semanticSHA, err := requiredString(payload, "render_request_semantic_sha256")
		if err != nil {
			return "", "", 0, nil, err
		}
		return orchestration.RenderWorkflowName,
			"realsas:render:" + commandID,
			6 * time.Hour,
			orchestration.RenderWorkflowInput{
				CommandID:                   commandID,
				SubjectID:                   subjectID,
				ProductRevisionID:           revisionID,
				RenderRequestID:             renderID,
				RenderRequestSemanticSHA256: semanticSHA,
			}, nil
	default:
		return "", "", 0, nil, fmt.Errorf("unsupported command type %q", commandType)
	}
}

func requiredString(payload map[string]any, key string) (string, error) {
	value, ok := payload[key].(string)
	if !ok || value == "" {
		return "", fmt.Errorf("command payload missing %s", key)
	}
	return value, nil
}
