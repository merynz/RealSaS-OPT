package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestSubjectIdentityReadIsExistingOnlyAndArchivedSubjectsDisappear(t *testing.T) {
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

	slug := "subject-read-" + uuid.NewString()
	handler := (API{Pool: pool}).HandlerWithSubjectIdentityRead()

	missing := httptest.NewRecorder()
	handler.ServeHTTP(missing, httptest.NewRequest(http.MethodGet, "/v1/subjects/by-slug/"+slug, nil))
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing lookup: %d %s", missing.Code, missing.Body.String())
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM subjects WHERE slug=$1", slug).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("read lookup created subject: %d", count)
	}

	created, err := input.EnsureSubject(ctx, pool, input.SubjectRequest{
		Slug:        slug,
		DisplayName: "Subject identity read test",
		CreatedBy:   "ci",
	})
	if err != nil {
		t.Fatal(err)
	}

	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/subjects/by-slug/"+slug, nil))
	if response.Code != http.StatusOK {
		t.Fatalf("lookup: %d %s", response.Code, response.Body.String())
	}
	var identity input.SubjectIdentity
	if err := json.Unmarshal(response.Body.Bytes(), &identity); err != nil {
		t.Fatal(err)
	}
	if identity.Schema != "RealSaS.SubjectIdentity.v1" || identity.SubjectID != created.SubjectID || identity.Slug != slug || identity.DisplayName != "Subject identity read test" {
		t.Fatalf("identity drift: %+v", identity)
	}

	if _, err := pool.Exec(ctx, "UPDATE subjects SET archived_at=now() WHERE id=$1", created.SubjectID); err != nil {
		t.Fatal(err)
	}
	archived := httptest.NewRecorder()
	handler.ServeHTTP(archived, httptest.NewRequest(http.MethodGet, "/v1/subjects/by-slug/"+slug, nil))
	if archived.Code != http.StatusNotFound {
		t.Fatalf("archived lookup: %d %s", archived.Code, archived.Body.String())
	}
}
