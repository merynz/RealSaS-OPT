package httpapi

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRequireBearerFailsClosedWhenTokenIsNotConfigured(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	if requireBearer(w, req, func() string { return "" }, "PRODUCT_GATEWAY") {
		t.Fatal("unconfigured auth unexpectedly admitted request")
	}
	if w.Code != 503 {
		t.Fatalf("status=%d want=503", w.Code)
	}
}

func TestRequireBearerRejectsWrongTokenAndAcceptsExactToken(t *testing.T) {
	token := func() string { return "correct" }
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	w := httptest.NewRecorder()
	if requireBearer(w, req, token, "PRODUCT_GATEWAY") {
		t.Fatal("wrong token unexpectedly admitted request")
	}
	if w.Code != 401 {
		t.Fatalf("status=%d want=401", w.Code)
	}

	req = httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer correct")
	w = httptest.NewRecorder()
	if !requireBearer(w, req, token, "PRODUCT_GATEWAY") {
		t.Fatalf("exact token rejected: status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestRequireProductOwnerValidatesDigest(t *testing.T) {
	req := httptest.NewRequest("GET", "/v1/product/compiles", nil)
	w := httptest.NewRecorder()
	if _, ok := requireProductOwner(w, req); ok {
		t.Fatal("missing product owner unexpectedly admitted")
	}
	if w.Code != 401 {
		t.Fatalf("missing owner status=%d want=401", w.Code)
	}

	req = httptest.NewRequest("GET", "/v1/product/compiles", nil)
	req.Header.Set("X-RealSaS-Product-Owner", strings.Repeat("a", 64))
	w = httptest.NewRecorder()
	owner, ok := requireProductOwner(w, req)
	if !ok || owner != strings.Repeat("a", 64) {
		t.Fatalf("valid owner rejected: owner=%q status=%d", owner, w.Code)
	}
}

func TestOwnedProductGatewayPathDoesNotCaptureInternalProductTools(t *testing.T) {
	for _, path := range []string{
		"/v1/product/compiles",
		"/v1/product/compiles/cmp_0123456789abcdef0123456789abcdef",
		"/v1/product/studio/projects/stp_0123456789abcdef0123456789abcdef",
	} {
		if !isOwnedProductGatewayPath(path) {
			t.Fatalf("public gateway path not ownership scoped: %s", path)
		}
	}
	for _, path := range []string{
		"/v1/product/compile",
		"/v1/product/render",
		"/v1/products/compile",
		"/v1/operator/product/preflight",
	} {
		if isOwnedProductGatewayPath(path) {
			t.Fatalf("internal/operator path captured by public ownership middleware: %s", path)
		}
	}
}
