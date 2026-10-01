package orchestration

const (
	ControlTaskQueue = "realsas-control-v1"
	EngineTaskQueue  = "realsas-engine-v1"

	CompileWorkflowName = "realsas.compile.v1"
	RenderWorkflowName  = "realsas.render.v1"

	ResolveCompilePlanActivityName   = "platform.resolve_compile_plan.v1"
	BindReusedStageActivityName      = "platform.bind_reused_stage.v1"
	CommitStageResultActivityName    = "platform.commit_stage_result.v1"
	FinalizeCompileActivityName      = "platform.finalize_compile_attempt.v1"
	EngineExecuteStageActivityName   = "engine.execute_compile_stage.v1"
	ResolveRenderRequestActivityName = "platform.resolve_render_request.v1"
	EngineRenderTailActivityName     = "engine.render_runtime_tail.v1"
	CommitRenderOutputActivityName   = "platform.commit_render_output.v1"
)

type CompileWorkflowInput struct {
	CommandID       string `json:"command_id"`
	AttemptID       string `json:"attempt_id"`
	SubjectID       string `json:"subject_id"`
	EngineReleaseID string `json:"engine_release_id"`
	SubjectInputID  string `json:"subject_input_id"`
	TargetStageID   string `json:"target_stage_id"`
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

type EngineStageRequest struct {
	CommandID              string   `json:"command_id"`
	AttemptID              string   `json:"attempt_id"`
	SubjectID              string   `json:"subject_id"`
	EngineReleaseID        string   `json:"engine_release_id"`
	StageID                string   `json:"stage_id"`
	ExpectedSemanticSHA256 string   `json:"expected_semantic_sha256"`
	AllowedExecuteStageIDs []string `json:"allowed_execute_stage_ids"`
}

type EngineOutput struct {
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

type StageCommitRequest struct {
	CommandID              string            `json:"command_id"`
	AttemptID              string            `json:"attempt_id"`
	StageID                string            `json:"stage_id"`
	ExpectedSemanticSHA256 string            `json:"expected_semantic_sha256"`
	EngineResult           EngineStageResult `json:"engine_result"`
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
	OutputPath    string         `json:"output_path"`
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
