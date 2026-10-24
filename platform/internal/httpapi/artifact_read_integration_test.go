package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
)

func TestArtifactHTTPReadSurfaceIsTypedBoundedAndHashVerified(t *testing.T) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	store, err := artifactstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	data := []byte(`{"schema":"RealSaS.HTTPReadProbe.v1","value":7}`)
	object, err := store.PutBytes(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	imported, err := registry.Import(ctx, pool, store, registry.ImportRequest{
		ArtifactType:  "RealSaS.HTTPReadProbe",
		SchemaVersion: "v1",
		Object:        object,
		SourceURI:     "urn:test:http-read:" + uuid.NewString(),
		CreatedBy:     "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	subject, err := input.EnsureSubject(ctx, pool, input.SubjectRequest{
		Slug:        "http-read-" + uuid.NewString(),
		DisplayName: "HTTP artifact read test",
		CreatedBy:   "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := input.Seal(ctx, pool, subject.SubjectID, []input.Binding{{Role: "manifest:http-read", ArtifactID: imported.ArtifactID}}, "ci"); err != nil {
		t.Fatal(err)
	}

	handler := (API{Pool: pool, Store: store}).Handler()
	metadataResponse := httptest.NewRecorder()
	handler.ServeHTTP(metadataResponse, httptest.NewRequest(http.MethodGet, "/v1/artifacts/"+imported.ArtifactID.String(), nil))
	if metadataResponse.Code != http.StatusOK {
		t.Fatalf("metadata: %d %s", metadataResponse.Code, metadataResponse.Body.String())
	}
	var metadata registry.ArtifactRecord
	if err := json.Unmarshal(metadataResponse.Body.Bytes(), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.ID != imported.ArtifactID || metadata.ArtifactType != "RealSaS.HTTPReadProbe" || metadata.ContentSHA256 != object.ContentSHA256 {
		t.Fatalf("metadata drift: %+v", metadata)
	}

	bytesResponse := httptest.NewRecorder()
	handler.ServeHTTP(bytesResponse, httptest.NewRequest(http.MethodGet, "/v1/artifacts/"+imported.ArtifactID.String()+"/bytes", nil))
	if bytesResponse.Code != http.StatusOK || !bytes.Equal(bytesResponse.Body.Bytes(), data) {
		t.Fatalf("bytes: %d %q", bytesResponse.Code, bytesResponse.Body.Bytes())
	}
	if bytesResponse.Header().Get("X-Content-SHA256") != object.ContentSHA256 || bytesResponse.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("identity headers drift: %v", bytesResponse.Header())
	}

	listPath := "/v1/subjects/" + subject.SubjectID.String() + "/artifacts?artifact_type=" + url.QueryEscape("RealSaS.HTTPReadProbe") + "&schema_version=v1&limit=10"
	listResponse := httptest.NewRecorder()
	handler.ServeHTTP(listResponse, httptest.NewRequest(http.MethodGet, listPath, nil))
	if listResponse.Code != http.StatusOK {
		t.Fatalf("list: %d %s", listResponse.Code, listResponse.Body.String())
	}
	var bindings []registry.SubjectArtifactRecord
	if err := json.Unmarshal(listResponse.Body.Bytes(), &bindings); err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 1 || bindings[0].ArtifactID != imported.ArtifactID || bindings[0].SourceKind != "SUBJECT_INPUT" {
		t.Fatalf("bindings drift: %+v", bindings)
	}

	var typeID uuid.UUID
	if err := pool.QueryRow(ctx, "SELECT artifact_type_id FROM artifacts WHERE id=$1", imported.ArtifactID).Scan(&typeID); err != nil {
		t.Fatal(err)
	}
	oversizedID := uuid.New()
	semantic := artifactstore.HashBytes([]byte("oversized-semantic-" + oversizedID.String()))
	content := artifactstore.HashBytes([]byte("oversized-content-" + oversizedID.String()))
	implementation := artifactstore.HashBytes([]byte("oversized-implementation"))
	policy := artifactstore.HashBytes([]byte("oversized-policy"))
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
		                      producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,$6,'http-read-size-guard',$7,$8,'{}'::jsonb,now())
	`, oversizedID, typeID, semantic, content, "cas/test/oversized-"+oversizedID.String(), maxArtifactReadBytes+1, implementation, policy); err != nil {
		t.Fatal(err)
	}
	oversizedResponse := httptest.NewRecorder()
	handler.ServeHTTP(oversizedResponse, httptest.NewRequest(http.MethodGet, "/v1/artifacts/"+oversizedID.String()+"/bytes", nil))
	if oversizedResponse.Code != http.StatusRequestEntityTooLarge || !bytes.Contains(oversizedResponse.Body.Bytes(), []byte("ARTIFACT_READ_SIZE_LIMIT_EXCEEDED")) {
		t.Fatalf("oversized read: %d %s", oversizedResponse.Code, oversizedResponse.Body.String())
	}
}
