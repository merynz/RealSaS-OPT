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

// withProductOwner is intentionally scoped to the narrow service-token product
// namespace. Developer/operator routes keep their existing identities. The
// digest is placed in context so lower product layers can enforce ownership,
// while compile/project subroutes are pre-authorized before any state is read.
func (a API) withProductOwner(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/product/") {
			next.ServeHTTP(w, r)
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
