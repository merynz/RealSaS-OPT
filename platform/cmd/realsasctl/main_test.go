package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
)

func TestArtifactPutOnlySendsExactBytes(t *testing.T) {
	data := []byte("exact evidence")
	sha := artifactstore.HashBytes(data)
	path := filepath.Join(t.TempDir(), "evidence")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != string(data) || r.Method != "PUT" || r.URL.Path != "/v1/artifacts/bytes" || r.URL.Query().Get("sha256") != sha || r.Header.Get("Content-Type") != "application/octet-stream" {
			t.Errorf("upload identity drift: %s %s %v", r.Method, r.URL, err)
		}
		w.WriteHeader(202)
	}))
	defer server.Close()
	args := []string{"artifact-put", "--api", server.URL, "--file", path, "--sha256", sha}
	if err := run(args); err != nil {
		t.Fatal(err)
	}
	args[len(args)-1] = artifactstore.HashBytes([]byte("wrong bytes"))
	if err := run(args); err == nil {
		t.Fatal("tampered hash sent to server")
	}
	args[len(args)-1] = "not-a-sha"
	if err := run(args); err == nil {
		t.Fatal("invalid hash sent to server")
	}
	if calls != 1 {
		t.Fatalf("calls=%d", calls)
	}
}
