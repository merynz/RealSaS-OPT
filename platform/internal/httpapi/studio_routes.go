package httpapi

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/merynz/RealSaS-OPT/platform/internal/studio"
)

// HandlerWithStudio adds the public Studio persistence lane around the product
// gateway. The browser continues to see only opaque public compile/project ids;
// ProductRevision and authoring revision UUIDs remain internal to Platform.
func (a API) HandlerWithStudio() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/product/compiles/{id}/studio-project", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		var req struct {
			DisplayName string `json:"display_name,omitempty"`
			CreatedBy   string `json:"created_by,omitempty"`
		}
		if !decode(w, r, &req) {
			return
		}
		out, err := studio.EnsureProject(r.Context(), a.Pool, r.PathValue("id"), req.DisplayName, req.CreatedBy)
		if errors.Is(err, studio.ErrCompileNotReady) {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, out)
	})

	mux.HandleFunc("GET /v1/product/studio/projects/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		out, err := studio.GetProject(r.Context(), a.Pool, r.PathValue("id"))
		if errors.Is(err, studio.ErrProjectNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, out)
	})

	mux.HandleFunc("GET /v1/product/studio/projects/{id}/head", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		out, err := studio.GetHead(r.Context(), a.Pool, r.PathValue("id"))
		if errors.Is(err, studio.ErrProjectNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, out)
	})

	mux.HandleFunc("GET /v1/product/studio/projects/{id}/revisions", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		limit := 50
		if raw := r.URL.Query().Get("limit"); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed <= 0 {
				respond(w, http.StatusBadRequest, map[string]string{"error": "STUDIO_REVISION_LIMIT_INVALID"})
				return
			}
			limit = parsed
		}
		out, err := studio.ListRevisions(r.Context(), a.Pool, r.PathValue("id"), limit)
		if errors.Is(err, studio.ErrProjectNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		respond(w, http.StatusOK, map[string]any{"revisions": out})
	})

	mux.HandleFunc("POST /v1/product/studio/projects/{id}/revisions", func(w http.ResponseWriter, r *http.Request) {
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		var req studio.SaveRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := studio.SaveRevision(r.Context(), a.Pool, r.PathValue("id"), req)
		switch {
		case errors.Is(err, studio.ErrProjectNotFound):
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		case errors.Is(err, studio.ErrRevisionConflict):
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
		case errors.Is(err, studio.ErrAuthoringPayloadInvalid):
			respond(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		case err != nil:
			respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
		default:
			respond(w, http.StatusCreated, out)
		}
	})

	mux.Handle("/", a.HandlerWithProductWorkspace())
	return mux
}
