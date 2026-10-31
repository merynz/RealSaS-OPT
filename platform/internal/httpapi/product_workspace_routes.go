package httpapi

import (
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/publicproduct"
)

// HandlerWithProductWorkspace adds the post-proof derived workspace lane around
// the existing product/capability API. The browser never receives revision or
// artifact identities; only the trusted edge may fetch the ready workspace zip.
func (a API) HandlerWithProductWorkspace() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/operator/product/workspace/claim", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		var req struct {
			Owner        string `json:"owner"`
			LeaseSeconds int64  `json:"lease_seconds,omitempty"`
		}
		if !decode(w, r, &req) {
			return
		}
		if req.LeaseSeconds == 0 {
			req.LeaseSeconds = 900
		}
		out, err := publicproduct.ClaimWorkspaceJob(
			r.Context(), a.Pool, req.Owner, time.Duration(req.LeaseSeconds)*time.Second,
		)
		if errors.Is(err, publicproduct.ErrNoWorkspaceJob) {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		result(w, out, err)
	})

	mux.HandleFunc("GET /v1/operator/product/compiles/{id}/workspace-source/{role}", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		leaseToken, err := uuid.Parse(r.Header.Get("X-RealSaS-Lease-Token"))
		if err != nil {
			respond(w, http.StatusBadRequest, map[string]string{"error": "valid X-RealSaS-Lease-Token required"})
			return
		}
		payload, item, err := publicproduct.ReadWorkspaceArtifactBytes(
			r.Context(), a.Pool, a.Store, r.PathValue("id"), leaseToken, r.PathValue("role"),
		)
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-RealSaS-Artifact-Type", item.ArtifactType)
		w.Header().Set("X-RealSaS-Schema-Version", item.SchemaVersion)
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	})

	mux.HandleFunc("POST /v1/operator/product/compiles/{id}/workspace-bundle", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		leaseToken, err := uuid.Parse(r.Header.Get("X-RealSaS-Lease-Token"))
		if err != nil {
			respond(w, http.StatusBadRequest, map[string]string{"error": "valid X-RealSaS-Lease-Token required"})
			return
		}
		actor := r.Header.Get("X-RealSaS-Worker")
		if actor == "" {
			respond(w, http.StatusBadRequest, map[string]string{"error": "X-RealSaS-Worker required"})
			return
		}
		reader := http.MaxBytesReader(w, r.Body, publicproduct.MaxWorkspaceBundleBytes+1)
		payload, err := io.ReadAll(reader)
		if err != nil {
			respond(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if err := publicproduct.CompleteWorkspaceBundleBytes(
			r.Context(), a.Pool, a.Store, r.PathValue("id"), leaseToken, payload, actor,
		); err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, map[string]string{"status": "PASS"})
	})

	mux.HandleFunc("POST /v1/operator/product/compiles/{id}/workspace-fail", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.OperatorToken, "OPERATOR") {
			return
		}
		var req struct {
			LeaseToken uuid.UUID `json:"lease_token"`
			ErrorCode  string    `json:"error_code"`
			Retry      bool      `json:"retry"`
		}
		if !decode(w, r, &req) {
			return
		}
		if err := publicproduct.FailWorkspaceJob(
			r.Context(), a.Pool, r.PathValue("id"), req.LeaseToken, req.ErrorCode, req.Retry,
		); err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, map[string]any{"status": "RECORDED", "retry": req.Retry})
	})

	mux.HandleFunc("GET /v1/product/compiles/{id}/workspace-bundle", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		payload, err := publicproduct.WorkspaceBundleBytes(r.Context(), a.Pool, a.Store, r.PathValue("id"))
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		w.Header().Set("Content-Type", "application/zip")
		w.Header().Set("Cache-Control", "private, no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	})

	mux.Handle("/", a.HandlerWithCapabilityRun())
	return mux
}
