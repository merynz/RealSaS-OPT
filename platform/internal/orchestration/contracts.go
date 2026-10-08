package orchestration

import "github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"

const (
	ControlTaskQueue = "realsas-control-v1"
	EngineTaskQueue  = "realsas-engine-v1"

	CompileWorkflowName    = "realsas.compile.v1"
	RenderWorkflowName     = "realsas.render.v1"
	CapabilityWorkflowName = "realsas.capability.v1"

	ResolveCompilePlanActivityName       = "platform.resolve_compile_plan.v1"
	PrepareStageExecutionActivityName    = "platform.prepare_stage_execution.v1"
	BindReusedStageActivityName          = "platform.bind_reused_stage.v1"
	RecordStageActivityErrorActivityName = "platform.record_stage_activity_error.v1"
	CommitStageResultActivityName        = "platform.commit_stage_result.v1"
	FinalizeCompileActivityName          = "platform.finalize_compile_attempt.v1"
	EngineExecuteStageActivityName       = "engine.execute_compile_stage.v1"
	ResolveRenderRequestActivityName     = "platform.resolve_render_request.v1"
	EngineRenderTailActivityName         = "engine.render_runtime_tail.v1"
	CommitRenderOutputActivityName       = "platform.commit_render_output.v1"

	ResolveCapabilityGoalActivityName         = "platform.resolve_capability_goal.v1"
	PrepareCapabilityExecutionActivityName    = "platform.prepare_capability_execution.v1"
	CommitCapabilityResultActivityName        = "platform.commit_capability_result.v1"
	RecordCapabilityActivityErrorActivityName = "platform.record_capability_activity_error.v1"
	FinalizeCapabilityGoalActivityName        = "platform.finalize_capability_goal.v1"
	EngineExecuteCapabilityActivityName       = "engine.execute_capability.v1"
)

type CompileWorkflowInput struct {
	ExecutionMode   string `json:"execution_mode,omitempty"`
	CommandID       string `json:"command_id"`
	AttemptID       string `json:"attempt_id"`
	SubjectID       string `json:"subject_id"`
	EngineReleaseID string `json:"engine_release_id"`
	SubjectInputID  string `json:"subject_input_id"`
	TargetStageID   string `json:"target_stage_id"`
}

type CapabilityWorkflowInput struct {
	CommandID       string `json:"command_id"`
	AttemptID       string `json:"attempt_id"`
	ExecutionGoalID string `json:"execution_goal_id"`
	EngineReleaseID string `json:"engine_release_id"`
	GoalSpecSHA256  string `json:"goal_spec_sha256"`
}

type CapabilityStep struct {
	CapabilityID         string         `json:"capability_id"`
	Kind                 string         `json:"kind"`
	OwnerModuleID        string         `json:"owner_module_id"`
	ExecutorActivity     string         `json:"executor_activity"`
	ImplementationSHA256 string         `json:"implementation_sha256"`
	PolicySHA256         string         `json:"policy_sha256"`
	ParametersSHA256     string         `json:"parameters_sha256"`
	Dependencies         []string       `json:"dependencies"`
	Metadata             map[string]any `json:"metadata,omitempty"`
}

type ResolvedCapabilityGoal struct {
	ExecutionGoalID     string           `json:"execution_goal_id"`
	GoalType            string           `json:"goal_type"`
	PromotionPolicy     string           `json:"promotion_policy"`
	CapabilitySetSHA256 string           `json:"capability_set_sha256"`
	Parameters          map[string]any   `json:"parameters"`
	Steps               []CapabilityStep `json:"steps"`
}

type ArtifactRef struct {
	ID             string `json:"id"`
	Role           string `json:"role"`
	ArtifactType   string `json:"artifact_type"`
	SchemaVersion  string `json:"schema_version"`
	SemanticSHA256 string `json:"semantic_sha256"`
	StorageKey     string `json:"storage_key"`
	ContentSHA256  string `json:"content_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

type PrepareCapabilityExecutionRequest struct {
	CommandID       string         `json:"command_id"`
	AttemptID       string         `json:"attempt_id"`
	ExecutionGoalID string         `json:"execution_goal_id"`
	EngineReleaseID string         `json:"engine_release_id"`
	Step            CapabilityStep `json:"step"`
	GoalParameters  map[string]any `json:"goal_parameters"`
}

type EngineCapabilityRequest struct {
	ReleasedGraph             *stagegraph.Snapshot `json:"released_graph,omitempty"`
	PipelinePlanSHA256        string               `json:"pipeline_plan_sha256,omitempty"`
	StageGraphNodeSHA256      string               `json:"stage_graph_node_sha256,omitempty"`
	StageSemanticParameters   map[string]any       `json:"stage_semantic_parameters,omitempty"`
	ExecutionID               string               `json:"execution_id"`
	CommandID                 string               `json:"command_id"`
	AttemptID                 string               `json:"attempt_id"`
	ExecutionGoalID           string               `json:"execution_goal_id"`
	EngineReleaseID           string               `json:"engine_release_id"`
	CapabilityID              string               `json:"capability_id"`
	Kind                      string               `json:"kind"`
	OwnerModuleID             string               `json:"owner_module_id"`
	RequestedExecutorActivity string               `json:"requested_executor_activity"`
	ImplementationSHA256      string               `json:"implementation_sha256"`
	PolicySHA256              string               `json:"policy_sha256"`
	ParametersSHA256          string               `json:"parameters_sha256"`
	CapabilityMetadata        map[string]any       `json:"capability_metadata,omitempty"`
	GoalParameters            map[string]any       `json:"goal_parameters"`
	InputArtifacts            []ArtifactRef        `json:"input_artifacts"`
}

type EngineCapabilityFailure struct {
	Code                  string         `json:"code"`
	Class                 string         `json:"class"`
	ReportedOwnerModuleID string         `json:"reported_owner_module_id,omitempty"`
	Diagnostics           map[string]any `json:"diagnostics,omitempty"`
}

type EngineCapabilityResult struct {
	CapabilityID string                   `json:"capability_id"`
	Status       string                   `json:"status"`
	Outputs      []EngineOutput           `json:"outputs"`
	Diagnostics  map[string]any           `json:"diagnostics,omitempty"`
	Failure      *EngineCapabilityFailure `json:"failure,omitempty"`
}

type CapabilityCommitRequest struct {
	Request EngineCapabilityRequest `json:"request"`
	Result  EngineCapabilityResult  `json:"result"`
}

type CapabilityCommitResult struct {
	CapabilityID      string   `json:"capability_id"`
	Status            string   `json:"status"`
	OutputArtifactIDs []string `json:"output_artifact_ids,omitempty"`
}

type CapabilityActivityErrorRequest struct {
	ExecutionID   string `json:"execution_id"`
	AttemptID     string `json:"attempt_id"`
	CapabilityID  string `json:"capability_id"`
	OwnerModuleID string `json:"owner_module_id"`
	Error         string `json:"error"`
}

type CapabilityWorkflowResult struct {
	AttemptID string                   `json:"attempt_id"`
	Status    string                   `json:"status"`
	Completed []CapabilityCommitResult `json:"completed"`
}

type RenderWorkflowInput struct {
	CommandID                   string `json:"command_id"`
	SubjectID                   string `json:"subject_id"`
	ProductRevisionID           string `json:"product_revision_id"`
	RenderRequestID             string `json:"render_request_id"`
	RenderRequestSemanticSHA256 string `json:"render_request_semantic_sha256"`
}

type ResolvedStage struct {
	StageID                string  `json:"stage_id"`
	Action                 string  `json:"action"`
	ExpectedSemanticSHA256 string  `json:"expected_semantic_sha256"`
	ReusableArtifactID     *string `json:"reusable_artifact_id,omitempty"`
	Reason                 string  `json:"reason"`
}

type ResolvedCompilePlan struct {
	TargetStageID string          `json:"target_stage_id"`
	Stages        []ResolvedStage `json:"stages"`
}

type PrepareStageExecutionRequest struct {
	CommandID              string   `json:"command_id"`
	AttemptID              string   `json:"attempt_id"`
	SubjectID              string   `json:"subject_id"`
	SubjectInputID         string   `json:"subject_input_id"`
	EngineReleaseID        string   `json:"engine_release_id"`
	StageID                string   `json:"stage_id"`
	ExpectedSemanticSHA256 string   `json:"expected_semantic_sha256"`
	AllowedExecuteStageIDs []string `json:"allowed_execute_stage_ids"`
}

type StageInput struct {
	StageID  string      `json:"stage_id"`
	Artifact ArtifactRef `json:"artifact"`
}

type EngineStageRequest struct {
	ReleasedGraph          stagegraph.Snapshot `json:"released_graph"`
	GraphNodeSHA256        string              `json:"graph_node_sha256"`
	SemanticParameters     map[string]any      `json:"semantic_parameters"`
	ExecutionMode          string              `json:"execution_mode"`
	ImplementationSHA256   string              `json:"implementation_sha256"`
	PolicySHA256           string              `json:"policy_sha256"`
	InputStages            []StageInput        `json:"input_stages"`
	SourceInputs           []ArtifactRef       `json:"source_inputs"`
	ExecutionID            string              `json:"execution_id"`
	CommandID              string              `json:"command_id"`
	AttemptID              string              `json:"attempt_id"`
	SubjectID              string              `json:"subject_id"`
	EngineReleaseID        string              `json:"engine_release_id"`
	CompilerRunID          string              `json:"compiler_run_id"`
	PipelinePlanSHA256     string              `json:"pipeline_plan_sha256"`
	StageID                string              `json:"stage_id"`
	ExpectedSemanticSHA256 string              `json:"expected_semantic_sha256"`
	AllowedExecuteStageIDs []string            `json:"allowed_execute_stage_ids"`
}

type EngineOutput struct {
	RelativePath   string `json:"relative_path,omitempty"`
	PayloadSchema  string `json:"payload_schema,omitempty"`
	Role           string `json:"role"`
	ArtifactType   string `json:"artifact_type"`
	SchemaVersion  string `json:"schema_version"`
	StorageKey     string `json:"storage_key"`
	ContentSHA256  string `json:"content_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
	AuthorityClass string `json:"authority_class"`
}

type EngineFailureEvidence struct {
	Code                 string         `json:"code"`
	Class                string         `json:"class"`
	ReportedOwnerStageID string         `json:"reported_owner_stage_id,omitempty"`
	Diagnostics          map[string]any `json:"diagnostics"`
}

type EngineStageResult struct {
	StageID          string                 `json:"stage_id"`
	Status           string                 `json:"status"`
	ExecutedStageIDs []string               `json:"executed_stage_ids"`
	Outputs          []EngineOutput         `json:"outputs"`
	DiagnosticsHash  string                 `json:"diagnostics_hash"`
	Failure          *EngineFailureEvidence `json:"failure,omitempty"`
}

type BindReusedStageRequest struct {
	AttemptID              string `json:"attempt_id"`
	StageID                string `json:"stage_id"`
	ExpectedSemanticSHA256 string `json:"expected_semantic_sha256"`
	ArtifactID             string `json:"artifact_id"`
}

type StageCommitRequest struct {
	ExecutionID            string            `json:"execution_id"`
	CommandID              string            `json:"command_id"`
	AttemptID              string            `json:"attempt_id"`
	StageID                string            `json:"stage_id"`
	ExpectedSemanticSHA256 string            `json:"expected_semantic_sha256"`
	AllowedExecuteStageIDs []string          `json:"allowed_execute_stage_ids"`
	EngineResult           EngineStageResult `json:"engine_result"`
}

type StageActivityErrorRequest struct {
	ExecutionID string `json:"execution_id"`
	AttemptID   string `json:"attempt_id"`
	StageID     string `json:"stage_id"`
	Error       string `json:"error"`
}

type StageCommitResult struct {
	StageID    string  `json:"stage_id"`
	Status     string  `json:"status"`
	ArtifactID *string `json:"artifact_id,omitempty"`
}

type CompileWorkflowResult struct {
	AttemptID         string              `json:"attempt_id"`
	Status            string              `json:"status"`
	CompletedStages   []StageCommitResult `json:"completed_stages"`
	ProductRevisionID *string             `json:"product_revision_id,omitempty"`
}

type RenderResolution struct {
	CacheHit         bool           `json:"cache_hit"`
	OutputArtifactID *string        `json:"output_artifact_id,omitempty"`
	RuntimeRequest   map[string]any `json:"runtime_request,omitempty"`
}

type EngineRenderResult struct {
	Status        string         `json:"status"`
	OutputPath    string         `json:"output_path,omitempty"`
	StorageKey    string         `json:"storage_key,omitempty"`
	ContentSHA256 string         `json:"content_sha256"`
	SizeBytes     int64          `json:"size_bytes"`
	Diagnostics   map[string]any `json:"diagnostics,omitempty"`
}

type RenderCommitRequest struct {
	CommandID       string             `json:"command_id"`
	RenderRequestID string             `json:"render_request_id"`
	EngineResult    EngineRenderResult `json:"engine_result"`
}

type RenderWorkflowResult struct {
	Status           string `json:"status"`
	CacheHit         bool   `json:"cache_hit"`
	OutputArtifactID string `json:"output_artifact_id"`
}
