package studio

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestStudioDBCourt(t *testing.T) {
	if os.Getenv("REALSAS_STUDIO_DB_COURT") != "1" {
		t.Skip("REALSAS_STUDIO_DB_COURT not enabled")
	}
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Fatal("REALSAS_DATABASE_URL is required for Studio DB court")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// The shipping product DB court runs immediately before this court and
	// leaves one real proof-qualified ProductRevision fixture in the same
	// ephemeral Postgres service. Reuse that authority instead of inventing a
	// weaker Studio-only fake ProductRevision graph.
	var productRevisionID, subjectID, attemptID, engineReleaseID uuid.UUID
	var manifestSHA string
	if err := pool.QueryRow(ctx, `
		SELECT pr.id, pr.subject_id, pr.created_from_attempt_id, a.engine_release_id, pr.manifest_sha256
		FROM product_revisions pr
		JOIN attempts a ON a.id=pr.created_from_attempt_id
		ORDER BY pr.sealed_at DESC
		LIMIT 1
	`).Scan(&productRevisionID, &subjectID, &attemptID, &engineReleaseID, &manifestSHA); err != nil {
		t.Fatalf("shipping ProductRevision fixture unavailable: %v", err)
	}
	if len(manifestSHA) != 64 {
		t.Fatalf("invalid ProductRevision manifest sha %q", manifestSHA)
	}

	var sourceArtifactID uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT a.id
		FROM artifacts a
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE t.name='RealSaS.SourceObservation' AND t.schema_version='v1'
		ORDER BY a.verified_at DESC NULLS LAST
		LIMIT 1
	`).Scan(&sourceArtifactID); err != nil {
		t.Fatalf("shipping SourceObservation fixture unavailable: %v", err)
	}
	var bundleArtifactID uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT artifact_id
		FROM product_revision_artifacts
		WHERE product_revision_id=$1
		ORDER BY role
		LIMIT 1
	`, productRevisionID).Scan(&bundleArtifactID); err != nil {
		t.Fatalf("shipping ProductRevision artifact unavailable: %v", err)
	}

	sessionID := uuid.New()
	compileID := "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := pool.Exec(ctx, `
		INSERT INTO public_product_compile_sessions(
			id, public_id, subject_id, engine_release_id, release_channel_lock_version,
			source_observation_artifact_id, attempt_id, product_revision_id,
			state, current_capability
		) VALUES ($1,$2,$3,$4,0,$5,$6,$7,'PASS','product.proof')
	`, sessionID, compileID, subjectID, engineReleaseID, sourceArtifactID, attemptID, productRevisionID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO public_product_workspace_materializations(
			session_id, product_revision_id, state, bundle_artifact_id
		) VALUES ($1,$2,'PASS',$3)
	`, sessionID, productRevisionID, bundleArtifactID); err != nil {
		t.Fatal(err)
	}

	project, err := EnsureProject(ctx, pool, compileID, "Studio DB Court", "studio-db-court")
	if err != nil {
		t.Fatal(err)
	}
	if !projectIDPattern.MatchString(project.ProjectID) || project.CompileID != compileID {
		t.Fatalf("unexpected Studio project identity: %+v", project)
	}
	if project.BaseManifestSHA256 != manifestSHA || project.LockVersion != 0 || project.HeadRevisionNumber != 0 {
		t.Fatalf("unexpected new Studio project state: %+v", project)
	}

	payload1 := studioDBCourtLayer(t, manifestSHA, "first")
	first, err := SaveRevision(ctx, pool, project.ProjectID, SaveRequest{
		ExpectedLockVersion: 0,
		AuthoringPayload:    payload1,
		CreatedBy:           "studio-db-court",
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.RevisionNumber != 1 || first.LockVersion != 1 || first.Idempotent || !first.RequiresRequalification {
		t.Fatalf("unexpected first revision: %+v", first)
	}

	// Repeating the same canonical authoring state is idempotent and must not
	// manufacture history or consume another optimistic lock version.
	repeated, err := SaveRevision(ctx, pool, project.ProjectID, SaveRequest{
		ExpectedLockVersion: 1,
		AuthoringPayload:    payload1,
		CreatedBy:           "studio-db-court",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !repeated.Idempotent || repeated.RevisionNumber != 1 || repeated.LockVersion != 1 {
		t.Fatalf("idempotent save drift: %+v", repeated)
	}

	payload2 := studioDBCourtLayer(t, manifestSHA, "second")
	second, err := SaveRevision(ctx, pool, project.ProjectID, SaveRequest{
		ExpectedLockVersion: 1,
		AuthoringPayload:    payload2,
		CreatedBy:           "studio-db-court",
	})
	if err != nil {
		t.Fatal(err)
	}
	if second.RevisionNumber != 2 || second.LockVersion != 2 || second.Idempotent {
		t.Fatalf("unexpected second revision: %+v", second)
	}

	// A stale editor/tab may not silently overwrite a newer head.
	if _, err := SaveRevision(ctx, pool, project.ProjectID, SaveRequest{
		ExpectedLockVersion: 1,
		AuthoringPayload:    studioDBCourtLayer(t, manifestSHA, "stale"),
		CreatedBy:           "stale-editor",
	}); !errors.Is(err, ErrRevisionConflict) {
		t.Fatalf("expected stale lock conflict, got %v", err)
	}

	head, err := GetHead(ctx, pool, project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if head.RevisionNumber != 2 || head.LockVersion != 2 || head.PayloadSHA256 != second.PayloadSHA256 {
		t.Fatalf("head drift after saves: %+v", head)
	}
	var decoded map[string]any
	if err := json.Unmarshal(head.AuthoringPayload, &decoded); err != nil {
		t.Fatal(err)
	}
	metadata, _ := decoded["metadata"].(map[string]any)
	if metadata["court_marker"] != "second" {
		t.Fatalf("head payload did not reload latest revision: %+v", metadata)
	}

	history, err := ListRevisions(ctx, pool, project.ProjectID, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 2 || history[0].RevisionNumber != 2 || !history[0].IsHead || history[1].RevisionNumber != 1 || history[1].IsHead {
		t.Fatalf("revision history drift: %+v", history)
	}

	// Append-only is enforced by Postgres itself, not just by Go repository
	// convention. Any in-place rewrite of a StudioAuthoringRevision must fail.
	if _, err := pool.Exec(ctx, `
		UPDATE studio_authoring_revisions
		SET created_by='illegal-rewrite'
		WHERE project_id=(SELECT id FROM studio_projects WHERE public_id=$1)
	`, project.ProjectID); err == nil || !strings.Contains(err.Error(), "STUDIO_AUTHORING_REVISION_IMMUTABLE") {
		t.Fatalf("immutable revision trigger did not reject update: %v", err)
	}
}

func studioDBCourtLayer(t *testing.T, stateHash, marker string) json.RawMessage {
	t.Helper()
	value := map[string]any{
		"schema_version":                    AuthoringLayerSchema,
		"source_product_state_hash":         stateHash,
		"authority_class":                   "OPTIONAL_USER_AUTHORING_LAYER",
		"can_promote_product_directly":      false,
		"requires_compiler_requalification": true,
		"metadata": map[string]any{
			"canonical_bundle_mutated": false,
			"court_marker":             marker,
		},
		"rig_pose_edits":     []any{},
		"mesh_vertex_edits":  []any{},
		"weight_edits":       []any{},
		"binding_edits":      []any{},
		"rig_topology_edits": []any{},
		"ik_constraints":     []any{},
		"animation_edits":    []any{},
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
