package httpapi

import (
	"crypto/subtle"
	"net/http"
	"os"
	"strings"
)

func (a API) ProductGatewayToken() string {
	return os.Getenv("REALSAS_PRODUCT_GATEWAY_TOKEN")
}

func (a API) OperatorToken() string {
	return os.Getenv("REALSAS_OPERATOR_TOKEN")
}

func requireBearer(w http.ResponseWriter, r *http.Request, expected func() string, scope string) bool {
	token := strings.TrimSpace(expected())
	if token == "" {
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
	if len(provided) != len(token) || subtle.ConstantTimeCompare([]byte(provided), []byte(token)) != 1 {
		respond(w, http.StatusUnauthorized, map[string]string{"error": "UNAUTHORIZED"})
		return false
	}
	return true
}
