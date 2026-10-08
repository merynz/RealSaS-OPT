package httpapi

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
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

func TestArtifactUploadRejectsDriftAndImportCannotSupplyQualification(t *testing.T) {
	store, err := artifactstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	data := "sealed bytes"
	sha := artifactstore.HashBytes([]byte(data))
	handler := (API{Store: store}).Handler()
	for _, tc := range []struct {
		method, path, body string
		status             int
	}{
		{"PUT", "/v1/artifacts/bytes?sha256=" + sha, "tampered", 409},
		{"PUT", "/v1/artifacts/bytes?sha256=invalid", data, 400},
		{"POST", "/v1/artifacts/import", `{"qualification":"REUSE_ELIGIBLE"}`, 400},
		{"PUT", "/v1/artifacts/bytes?sha256=" + sha, data, 202},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))
		if response.Code != tc.status {
			t.Fatalf("%s: %d %s", tc.path, response.Code, response.Body.String())
		}
	}
	key, _ := artifactstore.CASKey(sha, "")
	raw, err := store.GetBytes(context.Background(), key)
	if err != nil || string(raw) != data {
		t.Fatalf("CAS output drift: %v", err)
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
