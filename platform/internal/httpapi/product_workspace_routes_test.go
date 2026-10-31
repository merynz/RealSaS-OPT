package httpapi

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExactSourceObservationIngressIsOperatorOnly(t *testing.T) {
	t.Setenv("REALSAS_OPERATOR_TOKEN", "operator-secret")
	t.Setenv("REALSAS_PRODUCT_GATEWAY_TOKEN", "gateway-secret")
	handler := (API{}).HandlerWithProductWorkspace()

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/operator/product/compiles/from-source-observation",
		bytes.NewReader([]byte("source-observation")),
	)
	request.Header.Set("Authorization", "Bearer gateway-secret")
	request.Header.Set("X-RealSaS-Worker", "test")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("product gateway token reached operator ingress: status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestExactSourceObservationIngressRequiresOperatorIdentityBeforeDatabaseWork(t *testing.T) {
	t.Setenv("REALSAS_OPERATOR_TOKEN", "operator-secret")
	handler := (API{}).HandlerWithProductWorkspace()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/operator/product/compiles/from-source-observation",
		bytes.NewReader([]byte("source-observation")),
	)
	request.Header.Set("Authorization", "Bearer operator-secret")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("missing operator identity did not fail before DB work: status=%d body=%s", response.Code, response.Body.String())
	}
}
