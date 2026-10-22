package httpapi

import (
	"net/http"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/experiment"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelrelease"
	"github.com/merynz/RealSaS-OPT/platform/internal/operational"
)

// HandlerWithOperational layers P0 operational-truth routes over the existing
// API without weakening or duplicating the scientific execution handlers.
func (a API) HandlerWithOperational() http.Handler {
	mux := http.NewServeMux()

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
