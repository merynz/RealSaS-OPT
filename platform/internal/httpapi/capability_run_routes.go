package httpapi

import (
	"errors"
	"net/http"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	subjectinput "github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/publicproduct"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

type capabilityCompilerBridgeRequest struct {
	SubjectInputID     uuid.UUID `json:"subject_input_id"`
	CompilerRunID      string    `json:"compiler_run_id"`
	RunManifestPath    string    `json:"run_manifest_path"`
	RunLedgerPath      string    `json:"run_ledger_path"`
	PipelinePlanSHA256 string    `json:"pipeline_plan_sha256"`
}

type capabilityRunRequest struct {
	EngineReleaseID  uuid.UUID                        `json:"engine_release_id"`
	SubjectID        *uuid.UUID                       `json:"subject_id,omitempty"`
	Targets          []string                         `json:"targets"`
	InputArtifactIDs []uuid.UUID                      `json:"input_artifact_ids,omitempty"`
	Parameters       map[string]any                   `json:"parameters,omitempty"`
	CompilerBridge   *capabilityCompilerBridgeRequest `json:"compiler_bridge,omitempty"`
	IdempotencyKey   string                           `json:"idempotency_key"`
	RequestedBy      string                           `json:"requested_by"`
}

type productCapabilityCompileRequest struct {
	EngineReleaseID  uuid.UUID      `json:"engine_release_id"`
	SubjectID        uuid.UUID      `json:"subject_id"`
	InputArtifactIDs []uuid.UUID    `json:"input_artifact_ids"`
	Parameters       map[string]any `json:"parameters,omitempty"`
	IdempotencyKey   string         `json:"idempotency_key"`
	RequestedBy      string         `json:"requested_by"`
}

// HandlerWithCapabilityRun exposes developer/operator capability controls plus
// the narrow public product gateway. Public callers receive only opaque compile
// ids and product-facing state; release, subject, artifact, command, attempt and
// ProductRevision identities remain inside Platform.
func (a API) HandlerWithCapabilityRun() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/product/compiles", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		var req publicproduct.StartRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := publicproduct.Start(
			r.Context(), a.Pool, a.Store, a.CodeSHA, req, "public-product-gateway",
		)
		if errors.Is(err, publicproduct.ErrReleaseChannelUnbound) {
			respond(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/product/compiles/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		out, err := publicproduct.GetStatus(r.Context(), a.Pool, r.PathValue("id"))
		if errors.Is(err, publicproduct.ErrCompileNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, out)
	})
	mux.HandleFunc("POST /v1/operator/product-release-channel", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		var req struct {
			EngineReleaseID uuid.UUID `json:"engine_release_id"`
			Actor           string    `json:"actor"`
		}
		if !decode(w, r, &req) {
			return
		}
		if req.EngineReleaseID == uuid.Nil || req.Actor == "" {
			respond(w, http.StatusBadRequest, map[string]string{"error": "engine_release_id and actor are required"})
			return
		}
		out, err := publicproduct.BindProductionRelease(
			r.Context(), a.Pool, req.EngineReleaseID, a.CodeSHA, req.Actor,
		)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/operator/product/compiles/{id}/raw-views", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		out, err := publicproduct.RawViewArtifactIDs(r.Context(), a.Pool, r.PathValue("id"))
		if errors.Is(err, publicproduct.ErrCompileNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, map[string]any{"public_id": r.PathValue("id"), "raw_view_artifact_ids": out})
	})
	mux.HandleFunc("POST /v1/operator/product/compiles/{id}/source-observation", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		var req struct {
			ArtifactID uuid.UUID `json:"artifact_id"`
			Actor      string    `json:"actor"`
		}
		if !decode(w, r, &req) {
			return
		}
		if req.ArtifactID == uuid.Nil || req.Actor == "" {
			respond(w, http.StatusBadRequest, map[string]string{"error": "artifact_id and actor are required"})
			return
		}
		out, err := publicproduct.AttachSourceObservation(
			r.Context(), a.Pool, a.CodeSHA, r.PathValue("id"), req.ArtifactID, req.Actor,
		)
		if errors.Is(err, publicproduct.ErrCompileNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		result(w, out, err)
	})

	mux.HandleFunc("GET /v1/subjects/{subject}/inputs/{input}", func(w http.ResponseWriter, r *http.Request) {
		subjectID, err := uuid.Parse(r.PathValue("subject"))
		if err != nil {
			respond(w, http.StatusBadRequest, map[string]string{"error": "invalid subject id"})
			return
		}
		inputID, err := uuid.Parse(r.PathValue("input"))
		if err != nil {
			respond(w, http.StatusBadRequest, map[string]string{"error": "invalid subject input id"})
			return
		}
		loaded, err := subjectinput.Load(r.Context(), a.Pool, inputID, subjectID)
		if errors.Is(err, subjectinput.ErrEmpty) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, http.StatusOK, map[string]any{
			"schema":           "RealSaS.SubjectInputRead.v1",
			"subject_input_id": loaded.SubjectInputID,
			"subject_id":       loaded.SubjectID,
			"manifest_sha256":  loaded.ManifestSHA256,
			"root_inputs":      loaded.RootInputs,
			"artifact_ids":     loaded.ArtifactIDs,
		})
	})
	mux.HandleFunc("POST /v1/capabilities/run", func(w http.ResponseWriter, r *http.Request) {
		var req capabilityRunRequest
		if !decode(w, r, &req) {
			return
		}
		if req.EngineReleaseID == uuid.Nil || len(req.Targets) == 0 || req.IdempotencyKey == "" || req.RequestedBy == "" {
			respond(w, http.StatusBadRequest, map[string]string{"error": "engine_release_id, targets, idempotency_key and requested_by are required"})
			return
		}
		if err := release.RequireCanonicalMain(r.Context(), a.Pool, req.EngineReleaseID, a.CodeSHA); err != nil {
			result(w, nil, err)
			return
		}

		var bridge *command.CapabilityCompilerBridge
		if req.CompilerBridge != nil {
			bridge = &command.CapabilityCompilerBridge{
				SubjectInputID:     req.CompilerBridge.SubjectInputID,
				CompilerRunID:      req.CompilerBridge.CompilerRunID,
				RunManifestPath:    req.CompilerBridge.RunManifestPath,
				RunLedgerPath:      req.CompilerBridge.RunLedgerPath,
				PipelinePlanSHA256: req.CompilerBridge.PipelinePlanSHA256,
			}
		}
		out, err := command.SubmitCapabilityRun(r.Context(), a.Pool, command.CapabilityRunRequest{
			EngineReleaseID:  req.EngineReleaseID,
			SubjectID:        req.SubjectID,
			Targets:          append([]string(nil), req.Targets...),
			InputArtifactIDs: append([]uuid.UUID(nil), req.InputArtifactIDs...),
			Parameters:       req.Parameters,
			CompilerBridge:   bridge,
			IdempotencyKey:   req.IdempotencyKey,
			RequestedBy:      req.RequestedBy,
		})
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/products/compile", func(w http.ResponseWriter, r *http.Request) {
		var req productCapabilityCompileRequest
		if !decode(w, r, &req) {
			return
		}
		if req.EngineReleaseID == uuid.Nil || req.SubjectID == uuid.Nil || len(req.InputArtifactIDs) == 0 ||
			req.IdempotencyKey == "" || req.RequestedBy == "" {
			respond(w, http.StatusBadRequest, map[string]string{
				"error": "engine_release_id, subject_id, input_artifact_ids, idempotency_key and requested_by are required",
			})
			return
		}
		if err := release.RequireCanonicalMain(r.Context(), a.Pool, req.EngineReleaseID, a.CodeSHA); err != nil {
			result(w, nil, err)
			return
		}
		out, err := command.SubmitProductCapabilityCompile(r.Context(), a.Pool, command.ProductCapabilityCompileRequest{
			EngineReleaseID:  req.EngineReleaseID,
			SubjectID:        req.SubjectID,
			InputArtifactIDs: append([]uuid.UUID(nil), req.InputArtifactIDs...),
			Parameters:       req.Parameters,
			IdempotencyKey:   req.IdempotencyKey,
			RequestedBy:      req.RequestedBy,
		})
		result(w, out, err)
	})
	mux.Handle("/", a.HandlerWithSubjectIdentityRead())
	return mux
}
