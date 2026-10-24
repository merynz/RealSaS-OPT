package httpapi

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCapabilityRunRejectsIncompleteRequestBeforeDatabaseAccess(t *testing.T) {
	handler := (API{}).HandlerWithCapabilityRun()
	for _, body := range []string{
		`{}`,
		`{"engine_release_id":"00000000-0000-0000-0000-000000000000","targets":["product.rig"],"idempotency_key":"x","requested_by":"ci"}`,
		`{"unknown":true}`,
		`{} {}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest("POST", "/v1/capabilities/run", strings.NewReader(body)))
		if response.Code != 400 {
			t.Fatalf("body=%s status=%d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestCapabilityRunHandlerPreservesExistingReadRoutes(t *testing.T) {
	response := httptest.NewRecorder()
	(API{}).HandlerWithCapabilityRun().ServeHTTP(response, httptest.NewRequest("GET", "/health/live", nil))
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"status":"ok"`) {
		t.Fatalf("%d %s", response.Code, response.Body.String())
	}
}
