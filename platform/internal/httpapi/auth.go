package httpapi

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/publicproduct"
	"github.com/merynz/RealSaS-OPT/platform/internal/studio"
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

func requireProductOwner(w http.ResponseWriter, r *http.Request) (string, bool) {
	owner, err := publicproduct.NormalizeOwnerKeySHA256(r.Header.Get(publicproduct.ProductOwnerHeader))
	if err != nil {
		respond(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
		return "", false
	}
	return owner, true
}

func isOwnedProductGatewayPath(path string) bool {
	return path == "/v1/product/compiles" ||
		strings.HasPrefix(path, "/v1/product/compiles/") ||
		strings.HasPrefix(path, "/v1/product/studio/")
}

// withProductOwner is intentionally scoped to the narrow service-token public
// gateway, not the internal /v1/product/compile and /v1/product/render tools.
// Service authentication happens before DB ownership checks so unauthenticated
// callers cannot use owner probes as an existence oracle.
func (a API) withProductOwner(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isOwnedProductGatewayPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if !requireBearer(w, r, a.ProductGatewayToken, "PRODUCT_GATEWAY") {
			return
		}
		owner, ok := requireProductOwner(w, r)
		if !ok {
			return
		}
		ctx, err := publicproduct.WithOwnerKeySHA256(r.Context(), owner)
		if err != nil {
			respond(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
			return
		}

		const compilePrefix = "/v1/product/compiles/"
		if strings.HasPrefix(r.URL.Path, compilePrefix) {
			rest := strings.TrimPrefix(r.URL.Path, compilePrefix)
			compileID := strings.SplitN(rest, "/", 2)[0]
			if err := publicproduct.RequireOwner(ctx, a.Pool, compileID, owner); err != nil {
				if errors.Is(err, publicproduct.ErrCompileNotFound) {
					respond(w, http.StatusNotFound, map[string]string{"error": publicproduct.ErrCompileNotFound.Error()})
				} else {
					respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
				}
				return
			}
		}

		const projectPrefix = "/v1/product/studio/projects/"
		if strings.HasPrefix(r.URL.Path, projectPrefix) {
			rest := strings.TrimPrefix(r.URL.Path, projectPrefix)
			projectID := strings.SplitN(rest, "/", 2)[0]
			if err := studio.RequireProjectOwner(ctx, a.Pool, projectID, owner); err != nil {
				if errors.Is(err, studio.ErrProjectNotFound) {
					respond(w, http.StatusNotFound, map[string]string{"error": studio.ErrProjectNotFound.Error()})
				} else {
					respond(w, http.StatusConflict, map[string]string{"error": err.Error()})
				}
				return
			}
		}

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
