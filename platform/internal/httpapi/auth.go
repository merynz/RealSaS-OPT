package httpapi

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

func requireBearer(w http.ResponseWriter, r *http.Request, expected string, scope string) bool {
	expected = strings.TrimSpace(expected)
	if expected == "" {
		respond(w, http.StatusServiceUnavailable, map[string]string{
			"error": scope + "_AUTH_NOT_CONFIGURED",
		})
		return false
	}
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	const prefix = "Bearer "
	if !strings.HasPrefix(header, prefix) {
		respond(w, http.StatusUnauthorized, map[string]string{"error": "UNAUTHORIZED"})
		return false
	}
	provided := strings.TrimSpace(strings.TrimPrefix(header, prefix))
	if len(provided) != len(expected) || subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) != 1 {
		respond(w, http.StatusUnauthorized, map[string]string{"error": "UNAUTHORIZED"})
		return false
	}
	return true
}
