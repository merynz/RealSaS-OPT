package httpapi

import (
	"net/http"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/experiment"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelrelease"
	"github.com/merynz/RealSaS-OPT/platform/internal/operational"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

// HandlerWithOperational layers P0 operational-truth routes over the existing
// API without weakening or duplicating the scientific execution handlers.
func (a API) HandlerWithOperational() http.Handler {
	mux := http.NewServeMux()

	// EngineRelease source authority is minted only by the deployed Platform.
	// Clients can describe a release, but cannot self-assert that branch code is main.
	mux.HandleFunc("POST /v1/releases", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Manifest           release.Manifest                      `json:"manifest"`
			CapabilityVersions map[string]capability.VersionIdentity `json:"capability_versions"`
			CreatedBy          string                                `json:"created_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		out, err := release.Seal(r.Context(), a.Pool, a.Graph, req.Manifest, req.CreatedBy)
		if err != nil {
			result(w, out, err)
			return
		}
		if req.Manifest.ContractVersion == "RealSaS.EngineReleaseManifest.v2" {
			if _, err := capability.SealEngineReleaseSnapshot(r.Context(), a.Pool, out.ReleaseID, req.CapabilityVersions, req.CreatedBy); err != nil {
				result(w, out, err)
				return
			}
		}
		if _, err := release.BindCanonicalMain(r.Context(), a.Pool, out.ReleaseID, a.CodeSHA, req.CreatedBy); err != nil {
			result(w, out, err)
			return
		}
		result(w, out, nil)
	})
	mux.HandleFunc("GET /v1/releases/{id}/sources", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid release id"})
			return
		}
		out, err := release.SourceBindings(r.Context(), a.Pool, id)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/releases/{id}/capabilities", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid release id"})
			return
		}
		snapshot, err := capability.LoadReleaseSnapshot(r.Context(), a.Pool, id)
		if err != nil {
			result(w, nil, err)
			return
		}
		roles, projectionStatus, err := capability.ProductRoleReadProjection(snapshot.Registry)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, map[string]any{
			"schema":                         "RealSaS.ReleaseCapabilityGraph.v1",
			"engine_release_id":              snapshot.ReleaseID.String(),
			"capability_set_sha256":          snapshot.CapabilitySetSHA256,
			"capabilities":                   snapshot.Registry.Descriptors(),
			"versions":                       snapshot.Versions,
			"product_roles":                  roles,
			"product_role_projection_status": projectionStatus,
		})
	})
	mux.HandleFunc("POST /v1/research/attempts", func(w http.ResponseWriter, r *http.Request) {
		var req attempt.ResearchRequest
		if !decode(w, r, &req) {
			return
		}
		req.DeployedCodeSHA = a.CodeSHA
		out, err := attempt.StartResearch(r.Context(), a.Pool, a.Graph, req)
		result(w, out, err)
	})

	mux.HandleFunc("POST /v1/experiments", func(w http.ResponseWriter, r *http.Request) {
		var req experiment.CreateRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := experiment.Create(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/experiments/transition", func(w http.ResponseWriter, r *http.Request) {
		var req experiment.TransitionRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := experiment.Transition(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/experiments/link", func(w http.ResponseWriter, r *http.Request) {
		var req experiment.LinkRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := experiment.Link(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/experiments", func(w http.ResponseWriter, r *http.Request) {
		out, err := experiment.List(r.Context(), a.Pool)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/experiments/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid experiment id"})
			return
		}
		out, err := experiment.Get(r.Context(), a.Pool, id)
		if err != nil {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		respond(w, 200, out)
	})

	mux.HandleFunc("POST /v1/model-releases", func(w http.ResponseWriter, r *http.Request) {
		var req modelrelease.RegisterRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := modelrelease.Register(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/model-releases/evaluations", func(w http.ResponseWriter, r *http.Request) {
		var req modelrelease.EvaluationRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := modelrelease.SetEvaluation(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/model-releases/promote", func(w http.ResponseWriter, r *http.Request) {
		var req modelrelease.PromoteRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := modelrelease.Promote(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/model-releases", func(w http.ResponseWriter, r *http.Request) {
		out, err := modelrelease.List(r.Context(), a.Pool)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/model-releases/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid model release id"})
			return
		}
		out, err := modelrelease.Get(r.Context(), a.Pool, id)
		if err != nil {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/model-channels", func(w http.ResponseWriter, r *http.Request) {
		out, err := modelrelease.Channels(r.Context(), a.Pool)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/ops/status", func(w http.ResponseWriter, r *http.Request) {
		out, err := operational.Status(r.Context(), a.Pool)
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})

	mux.Handle("/", a.Handler())
	return mux
}
