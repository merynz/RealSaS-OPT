package modelrelease

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func sourceAuthorityHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func TestModelReleaseRequiresCanonicalMainSourceBinding(t *testing.T) {
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

	suffix := uuid.New().String()
	releaseID := uuid.New()
	artifactTypeID := uuid.New()
	artifactID := uuid.New()
	checkpointSHA := sourceAuthorityHash("checkpoint:" + suffix)
	legacySHA := strings.Repeat("a", 40)
	canonicalSHA := strings.Repeat("b", 40)

	if _, err := pool.Exec(ctx, `INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())`, releaseID, "model-source-"+suffix, sourceAuthorityHash("release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','model')", artifactTypeID, "model-source-checkpoint-"+suffix); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,
		 implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'ci',$6,$7,'{}',now())`, artifactID, artifactTypeID,
		sourceAuthorityHash("semantic:"+suffix), checkpointSHA, "model-source/"+suffix,
		sourceAuthorityHash("implementation:"+suffix), sourceAuthorityHash("policy:"+suffix)); err != nil {
		t.Fatal(err)
	}

	register := func(name, sourceSHA string) error {
		_, err := Register(ctx, pool, RegisterRequest{
			Family:                  "MIRA",
			Name:                    name + "-" + suffix,
			ArchitectureID:          "RealSaS.MIRA.SourceAuthority.Test.v1",
			CheckpointArtifactID:    artifactID,
			CheckpointContentSHA256: checkpointSHA,
			EngineReleaseID:         releaseID,
			SourceCodeSHA:           sourceSHA,
			InitialState:            StateDraft,
			CreatedBy:               "ci",
		})
		return err
	}

	if err := register("unbound", canonicalSHA); err == nil || !strings.Contains(err.Error(), "MODEL_RELEASE_CANONICAL_MAIN_SOURCE_REQUIRED") {
		t.Fatalf("unbound ModelRelease source was not rejected: %v", err)
	}

	if _, err := release.BindLegacyFrozen(ctx, pool, releaseID, legacySHA, "ci"); err != nil {
		t.Fatal(err)
	}
	if err := register("legacy", legacySHA); err == nil || !strings.Contains(err.Error(), "MODEL_RELEASE_CANONICAL_MAIN_SOURCE_REQUIRED") {
		t.Fatalf("LEGACY_FROZEN ModelRelease source was not rejected: %v", err)
	}

	if _, err := release.BindCanonicalMain(ctx, pool, releaseID, canonicalSHA, "ci"); err != nil {
		t.Fatal(err)
	}
	if err := register("canonical", canonicalSHA); err != nil {
		t.Fatalf("CANONICAL_MAIN ModelRelease source was rejected: %v", err)
	}
}
