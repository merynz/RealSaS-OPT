package product

import (
	"context"
	"testing"

	"github.com/google/uuid"
)

func TestProductRevisionRequiresImmutableCapabilitySnapshot(t *testing.T) {
	pool := integrationPool(t)
	ctx := context.Background()
	subjectID := uuid.New()
	attemptID := uuid.New()
	releaseID := uuid.New()
	revisionID := uuid.New()

	if _, err := pool.Exec(ctx,
		"INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)",
		subjectID, "wiring-"+subjectID.String(), "Release Wiring",
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, releaseID, "wiring-release-"+releaseID.String(), uuidHex64(releaseID)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','QUALIFIED')
	`, attemptID, subjectID, releaseID, uuidHex64(attemptID)); err != nil {
		t.Fatal(err)
	}

	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revisions(id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at)
		VALUES ($1,$2,1,$3,$4,now())
	`, revisionID, subjectID, uuidHex64(revisionID), attemptID); err == nil {
		t.Fatal("product revision sealed without capability snapshot")
	}

	capabilitySetSHA := uuidHex64(uuid.New())
	if _, err := pool.Exec(ctx,
		"UPDATE engine_releases SET capability_set_sha256=$2 WHERE id=$1",
		releaseID, capabilitySetSHA,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revisions(id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at)
		VALUES ($1,$2,1,$3,$4,now())
	`, revisionID, subjectID, uuidHex64(revisionID), attemptID); err != nil {
		t.Fatal(err)
	}

	var boundRelease uuid.UUID
	var boundCapability string
	if err := pool.QueryRow(ctx,
		"SELECT engine_release_id,capability_set_sha256 FROM product_revisions WHERE id=$1",
		revisionID,
	).Scan(&boundRelease, &boundCapability); err != nil {
		t.Fatal(err)
	}
	if boundRelease != releaseID || boundCapability != capabilitySetSHA {
		t.Fatalf("bound release=%s capability=%s", boundRelease, boundCapability)
	}

	if _, err := pool.Exec(ctx,
		"UPDATE product_revisions SET capability_set_sha256=$2 WHERE id=$1",
		revisionID, uuidHex64(uuid.New()),
	); err == nil {
		t.Fatal("product revision capability wiring mutated after seal")
	}
}
