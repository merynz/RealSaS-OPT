package httpapi

import (
	"errors"
	"net/http"

	"github.com/merynz/RealSaS-OPT/platform/internal/input"
)

// HandlerWithSubjectIdentityRead adds the narrow read-only subject identity
// projection needed by operators that must not call EnsureSubject. Subject
// creation and identity remain owned by the Input responsibility and database.
func (a API) HandlerWithSubjectIdentityRead() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/subjects/by-slug/{slug}", func(w http.ResponseWriter, r *http.Request) {
		out, err := input.LookupSubjectBySlug(r.Context(), a.Pool, r.PathValue("slug"))
		if errors.Is(err, input.ErrSubjectNotFound) {
			respond(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, http.StatusOK, out)
	})
	mux.Handle("/", a.HandlerWithOperational())
	return mux
}
