package httpapi

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLaneMismatchRejectedBeforeAnyDatabaseMutation(t *testing.T) {
	handler := (API{}).Handler()
	for _, tc := range []struct{ path, body string }{
		{"/v1/research/compile", `{}`},
		{"/v1/product/compile", `{"research_attempt_id":"4e5af44d-7b74-4a24-a720-de1a4eb99844"}`},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest("POST", tc.path, strings.NewReader(tc.body)))
		if response.Code != 400 || !strings.Contains(response.Body.String(), "EXECUTION_LANE_REQUEST_MISMATCH") {
			t.Fatalf("%s: %d %s", tc.path, response.Code, response.Body.String())
		}
	}
}

func TestRequestsRejectUnknownFieldsAndTrailingJSON(t *testing.T) {
	for _, body := range []string{`{"unknown":true}`, `{} {}`} {
		response := httptest.NewRecorder()
		(API{}).Handler().ServeHTTP(response, httptest.NewRequest("POST", "/v1/research/compile", strings.NewReader(body)))
		if response.Code != 400 {
			t.Fatalf("%d: %s", response.Code, response.Body.String())
		}
	}
}
