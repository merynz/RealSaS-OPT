package release

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func sourceAuthorityHash(prefix, suffix string) string {
	sum := sha256.Sum256([]byte(prefix + ":" + suffix))
	return hex.EncodeToString(sum[:])
}

func TestEngineReleaseCanonicalSourceAuthorityIsExactAndLegacyCannotRun(t *testing.T) {
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

	releaseID := uuid.New()
	suffix := uuid.NewString()
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())
	`, releaseID, "source-authority-"+suffix, sourceAuthorityHash("release", suffix)); err != nil {
		t.Fatal(err)
	}

	mainSHA := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	otherMainSHA := "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	legacySHA := "cccccccccccccccccccccccccccccccccccccccc"

	bound, err := BindCanonicalMain(ctx, pool, releaseID, mainSHA, "ci")
	if err != nil {
		t.Fatal(err)
	}
	if bound.Authority != SourceAuthorityCanonicalMain || bound.CodeSHA != mainSHA || bound.Reused {
		t.Fatalf("unexpected first canonical binding: %+v", bound)
	}
	boundAgain, err := BindCanonicalMain(ctx, pool, releaseID, mainSHA, "ci-2")
	if err != nil {
		t.Fatal(err)
	}
	if !boundAgain.Reused {
		t.Fatal("exact canonical source binding must be idempotent")
	}
	if err := RequireCanonicalMain(ctx, pool, releaseID, mainSHA); err != nil {
		t.Fatalf("canonical source rejected: %v", err)
	}
	if err := RequireCanonicalMain(ctx, pool, releaseID, otherMainSHA); err == nil {
		t.Fatal("different deployed main SHA must not inherit source authority")
	}

	if _, err := BindLegacyFrozen(ctx, pool, releaseID, legacySHA, "migration"); err != nil {
		t.Fatal(err)
	}
	if err := RequireCanonicalMain(ctx, pool, releaseID, legacySHA); err == nil {
		t.Fatal("LEGACY_FROZEN source must never satisfy canonical execution authority")
	}

	bindings, err := SourceBindings(ctx, pool, releaseID)
	if err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 2 {
		t.Fatalf("expected canonical + legacy provenance bindings, got %d", len(bindings))
	}

	var modelChannelCount int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM model_channels").Scan(&modelChannelCount); err != nil {
		t.Fatal(err)
	}
	var productPromotionCount int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM promotions").Scan(&productPromotionCount); err != nil {
		t.Fatal(err)
	}
	if modelChannelCount != 0 || productPromotionCount != 0 {
		t.Fatalf("source authority must not mint product authority: model_channels=%d promotions=%d", modelChannelCount, productPromotionCount)
	}
}
