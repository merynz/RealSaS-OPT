package httpapi

import (
	"net/http/httptest"
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
