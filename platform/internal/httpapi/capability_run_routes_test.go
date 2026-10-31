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

func TestProductCapabilityCompileRejectsIncompleteOrLegacyBridgeRequestBeforeDatabaseAccess(t *testing.T) {
	handler := (API{}).HandlerWithCapabilityRun()
	for _, body := range []string{
		`{}`,
		`{"engine_release_id":"00000000-0000-0000-0000-000000000000","subject_id":"00000000-0000-0000-0000-000000000000","idempotency_key":"x","requested_by":"ci"}`,
		`{"engine_release_id":"11111111-1111-1111-1111-111111111111","subject_id":"22222222-2222-2222-2222-222222222222","input_artifact_ids":[],"idempotency_key":"x","requested_by":"ci"}`,
		`{"engine_release_id":"11111111-1111-1111-1111-111111111111","subject_id":"22222222-2222-2222-2222-222222222222","input_artifact_ids":["33333333-3333-3333-3333-333333333333"],"compiler_bridge":{},"idempotency_key":"x","requested_by":"ci"}`,
		`{"unknown":true}`,
		`{} {}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest("POST", "/v1/products/compile", strings.NewReader(body)))
		if response.Code != 400 {
			t.Fatalf("body=%s status=%d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestSubjectInputReadRejectsInvalidIdentityBeforeDatabaseAccess(t *testing.T) {
	response := httptest.NewRecorder()
	(API{}).HandlerWithCapabilityRun().ServeHTTP(response, httptest.NewRequest("GET", "/v1/subjects/not-a-uuid/inputs/not-an-input", nil))
	if response.Code != 400 || !strings.Contains(response.Body.String(), "invalid subject id") {
		t.Fatalf("%d %s", response.Code, response.Body.String())
	}
}

func TestCapabilityRunHandlerPreservesExistingReadRoutes(t *testing.T) {
	response := httptest.NewRecorder()
	(API{}).HandlerWithCapabilityRun().ServeHTTP(response, httptest.NewRequest("GET", "/health/live", nil))
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"status":"ok"`) {
		t.Fatalf("%d %s", response.Code, response.Body.String())
	}
}
