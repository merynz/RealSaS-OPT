package studio

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
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
	ownerKey := strings.Repeat("a", 64)
	if _, err := pool.Exec(ctx, `
		INSERT INTO public_product_compile_sessions(
			id, public_id, subject_id, engine_release_id, release_channel_lock_version,
			owner_key_sha256, source_observation_artifact_id, attempt_id, product_revision_id,
			state, current_capability
		) VALUES ($1,$2,$3,$4,0,$5,$6,$7,$8,'PASS','product.proof')
	`, sessionID, compileID, subjectID, engineReleaseID, ownerKey, sourceArtifactID, attemptID, productRevisionID); err != nil {
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
	if err := RequireProjectOwner(ctx, pool, project.ProjectID, ownerKey); err != nil {
		t.Fatalf("correct Studio owner rejected: %v", err)
	}
	if err := RequireProjectOwner(ctx, pool, project.ProjectID, strings.Repeat("b", 64)); !errors.Is(err, ErrProjectNotFound) {
		t.Fatalf("cross-owner Studio lookup leaked project existence: %v", err)
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

	// Restore is append-only: selecting revision 1 creates revision 3 with the
	// old payload rather than rewinding the project head to revision 1.
	restored, err := RestoreRevision(ctx, pool, project.ProjectID, RestoreRequest{
		ExpectedLockVersion: 2,
		RevisionNumber:      1,
		CreatedBy:           "studio-db-court-restore",
	})
	if err != nil {
		t.Fatal(err)
	}
	if restored.RevisionNumber != 3 || restored.LockVersion != 3 || restored.Idempotent || restored.PayloadSHA256 != first.PayloadSHA256 {
		t.Fatalf("unexpected restored revision: %+v", restored)
	}
	if !restored.RequiresRequalification {
		t.Fatal("restored Studio revision must require requalification")
	}

	// A stale restore request must obey the same optimistic concurrency rule.
	if _, err := RestoreRevision(ctx, pool, project.ProjectID, RestoreRequest{
		ExpectedLockVersion: 2,
		RevisionNumber:      2,
		CreatedBy:           "stale-restore",
	}); !errors.Is(err, ErrRevisionConflict) {
		t.Fatalf("expected stale restore lock conflict, got %v", err)
	}

	head, err = GetHead(ctx, pool, project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if head.RevisionNumber != 3 || head.LockVersion != 3 || head.PayloadSHA256 != first.PayloadSHA256 {
		t.Fatalf("head drift after restore: %+v", head)
	}
	decoded = map[string]any{}
	if err := json.Unmarshal(head.AuthoringPayload, &decoded); err != nil {
		t.Fatal(err)
	}
	metadata, _ = decoded["metadata"].(map[string]any)
	if metadata["court_marker"] != "first" {
		t.Fatalf("restore did not recover revision 1 payload: %+v", metadata)
	}

	// Restoring the payload already at head is idempotent.
	restoredAgain, err := RestoreRevision(ctx, pool, project.ProjectID, RestoreRequest{
		ExpectedLockVersion: 3,
		RevisionNumber:      1,
		CreatedBy:           "studio-db-court-restore",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !restoredAgain.Idempotent || restoredAgain.RevisionNumber != 3 || restoredAgain.LockVersion != 3 {
		t.Fatalf("idempotent restore drift: %+v", restoredAgain)
	}

	history, err = ListRevisions(ctx, pool, project.ProjectID, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 3 || history[0].RevisionNumber != 3 || !history[0].IsHead || history[1].RevisionNumber != 2 || history[2].RevisionNumber != 1 {
		t.Fatalf("revision history drift after restore: %+v", history)
	}

	// Requalification starts only from the immutable saved HEAD. The first
	// shipping slice is one linear animator-control clip and no other edit domain.
	animationRevision, err := SaveRevision(ctx, pool, project.ProjectID, SaveRequest{
		ExpectedLockVersion: 3,
		AuthoringPayload:    studioDBRequalificationLayer(t, manifestSHA),
		CreatedBy:           "studio-db-court",
	})
	if err != nil {
		t.Fatal(err)
	}
	if animationRevision.RevisionNumber != 4 || animationRevision.LockVersion != 4 {
		t.Fatalf("unexpected animation revision: %+v", animationRevision)
	}
	var revisionCountBefore int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM product_revisions WHERE subject_id=$1`, subjectID).Scan(&revisionCountBefore); err != nil {
		t.Fatal(err)
	}
	store, err := artifactstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	requalification, err := RequestRequalification(ctx, pool, store, project.ProjectID, RequalificationRequest{
		ExpectedLockVersion: 4,
		RevisionNumber:      4,
		CreatedBy:           "studio-db-court",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(requalification.RequalificationID, "srq_") || requalification.State != "QUEUED" || requalification.Terminal {
		t.Fatalf("unexpected requalification admission: %+v", requalification)
	}
	if len(requalification.InvalidatedDomains) != 2 || requalification.InvalidatedDomains[0] != "MOTION" || requalification.InvalidatedDomains[1] != "PROOF" {
		t.Fatalf("requalification domains drift: %+v", requalification.InvalidatedDomains)
	}

	// Repeating the same HEAD request reuses the durable public request instead
	// of manufacturing a second Attempt or captured Studio artifact.
	requalificationAgain, err := RequestRequalification(ctx, pool, store, project.ProjectID, RequalificationRequest{
		ExpectedLockVersion: 4,
		RevisionNumber:      4,
		CreatedBy:           "studio-db-court-retry",
	})
	if err != nil {
		t.Fatal(err)
	}
	if requalificationAgain.RequalificationID != requalification.RequalificationID {
		t.Fatalf("requalification idempotency drift: %s != %s", requalificationAgain.RequalificationID, requalification.RequalificationID)
	}

	var requalificationAttemptID, authoringArtifactID uuid.UUID
	var requalificationKind string
	var parentAttemptID *uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT rq.attempt_id,rq.authoring_artifact_id,a.kind,a.parent_attempt_id
		FROM studio_requalification_requests rq
		JOIN attempts a ON a.id=rq.attempt_id
		WHERE rq.public_id=$1
	`, requalification.RequalificationID).Scan(
		&requalificationAttemptID, &authoringArtifactID, &requalificationKind, &parentAttemptID,
	); err != nil {
		t.Fatal(err)
	}
	if requalificationKind != "requalification_candidate" || parentAttemptID == nil || *parentAttemptID != attemptID {
		t.Fatalf("requalification attempt lineage drift: kind=%s parent=%v base=%s", requalificationKind, parentAttemptID, attemptID)
	}

	var capturedType, capturedSchema, capturedProducer string
	var capturedParams []byte
	if err := pool.QueryRow(ctx, `
		SELECT t.name,t.schema_version,a.producer_contract,a.semantic_parameters
		FROM artifacts a JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE a.id=$1
	`, authoringArtifactID).Scan(&capturedType, &capturedSchema, &capturedProducer, &capturedParams); err != nil {
		t.Fatal(err)
	}
	if capturedType != "RealSaS.StudioAuthoringRevision" || capturedSchema != "v1" || capturedProducer != "RealSaS.StudioAuthoringCapture.v1" {
		t.Fatalf("captured Studio artifact identity drift: %s %s %s", capturedType, capturedSchema, capturedProducer)
	}
	var captureMetadata map[string]any
	if err := json.Unmarshal(capturedParams, &captureMetadata); err != nil {
		t.Fatal(err)
	}
	if captureMetadata["authority_class"] != "STUDIO_AUTHORING_INPUT" || captureMetadata["qualification_granted"] != false || captureMetadata["can_promote_product_directly"] != false {
		t.Fatalf("captured Studio authority drift: %+v", captureMetadata)
	}
	var captureQualifications int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM qualifications WHERE artifact_id=$1`, authoringArtifactID).Scan(&captureQualifications); err != nil {
		t.Fatal(err)
	}
	if captureQualifications != 0 {
		t.Fatalf("Studio capture unexpectedly received qualification count=%d", captureQualifications)
	}

	for _, capabilityID := range animationRequalificationSatisfied {
		prefix := "capability:" + capabilityID + ":%"
		var inherited int
		if err := pool.QueryRow(ctx, `
			SELECT count(*) FROM attempt_artifacts
			WHERE attempt_id=$1 AND origin='inherited' AND role LIKE $2
		`, requalificationAttemptID, prefix).Scan(&inherited); err != nil {
			t.Fatal(err)
		}
		if inherited == 0 {
			t.Fatalf("missing inherited qualified authority for %s", capabilityID)
		}
	}
	var prematureExecutions int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM executions WHERE attempt_id=$1`, requalificationAttemptID).Scan(&prematureExecutions); err != nil {
		t.Fatal(err)
	}
	if prematureExecutions != 0 {
		t.Fatalf("requalification admission manufactured execution PASS state: count=%d", prematureExecutions)
	}

	var modelDrift bool
	if err := pool.QueryRow(ctx, `
		SELECT EXISTS(
			(SELECT family,model_release_id,selected_channel,selected_channel_lock_version
			 FROM attempt_model_releases WHERE attempt_id=$1
			 EXCEPT
			 SELECT family,model_release_id,selected_channel,selected_channel_lock_version
			 FROM attempt_model_releases WHERE attempt_id=$2)
			UNION ALL
			(SELECT family,model_release_id,selected_channel,selected_channel_lock_version
			 FROM attempt_model_releases WHERE attempt_id=$2
			 EXCEPT
			 SELECT family,model_release_id,selected_channel,selected_channel_lock_version
			 FROM attempt_model_releases WHERE attempt_id=$1)
		)
	`, attemptID, requalificationAttemptID).Scan(&modelDrift); err != nil {
		t.Fatal(err)
	}
	if modelDrift {
		t.Fatal("requalification attempt did not copy the exact base model snapshot")
	}

	var goalType string
	var resolvedRaw, parametersRaw []byte
	if err := pool.QueryRow(ctx, `
		SELECT goal_type,resolved_capability_ids,parameters
		FROM execution_goals WHERE attempt_id=$1
	`, requalificationAttemptID).Scan(&goalType, &resolvedRaw, &parametersRaw); err != nil {
		t.Fatal(err)
	}
	if goalType != "PRODUCT_REQUALIFICATION" {
		t.Fatalf("requalification goal type=%s", goalType)
	}
	var resolvedIDs []string
	var goalParameters map[string]any
	if err := json.Unmarshal(resolvedRaw, &resolvedIDs); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(parametersRaw, &goalParameters); err != nil {
		t.Fatal(err)
	}
	if len(resolvedIDs) != 9 || resolvedIDs[len(resolvedIDs)-1] != "product.proof" {
		t.Fatalf("requalification authority closure drift: %+v", resolvedIDs)
	}
	executionIDs, _ := goalParameters["execution_capability_ids"].([]any)
	if len(executionIDs) != 2 || executionIDs[0] != "product.motion" || executionIDs[1] != "product.proof" {
		t.Fatalf("requalification execution plan drift: %+v", goalParameters["execution_capability_ids"])
	}

	var revisionCountAfter int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM product_revisions WHERE subject_id=$1`, subjectID).Scan(&revisionCountAfter); err != nil {
		t.Fatal(err)
	}
	if revisionCountAfter != revisionCountBefore {
		t.Fatalf("requalification admission minted ProductRevision before Proof: %d -> %d", revisionCountBefore, revisionCountAfter)
	}
	var baseManifestAfter string
	if err := pool.QueryRow(ctx, `SELECT manifest_sha256 FROM product_revisions WHERE id=$1`, productRevisionID).Scan(&baseManifestAfter); err != nil {
		t.Fatal(err)
	}
	if baseManifestAfter != manifestSHA {
		t.Fatalf("base ProductRevision mutated during requalification admission: %s != %s", baseManifestAfter, manifestSHA)
	}

	// Requalification request identity is immutable in Postgres. Operational
	// state may advance, but base/edit/Attempt lineage may not be rewritten.
	if _, err := pool.Exec(ctx, `
		UPDATE studio_requalification_requests
		SET invalidated_domains='["PROOF"]'::jsonb
		WHERE public_id=$1
	`, requalification.RequalificationID); err == nil || !strings.Contains(err.Error(), "STUDIO_REQUALIFICATION_IDENTITY_IMMUTABLE") {
		t.Fatalf("requalification identity mutation was not rejected: %v", err)
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

	// Public ownership is also immutable at the database boundary. Account
	// linking/transfers must be an explicit future contract, never an UPDATE.
	if _, err := pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET owner_key_sha256=$2
		WHERE public_id=$1
	`, compileID, strings.Repeat("c", 64)); err == nil || !strings.Contains(err.Error(), "PUBLIC_PRODUCT_COMPILE_SESSION_IDENTITY_IMMUTABLE") {
		t.Fatalf("compile ownership mutation was not rejected: %v", err)
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
		"rig_pose_edits":        []any{},
		"mesh_vertex_edits":     []any{},
		"mesh_topology_edits":   []any{},
		"weight_edits":          []any{},
		"binding_edits":         []any{},
		"presentation_override": nil,
		"rig_topology_edits":    []any{},
		"ik_constraints":        []any{},
		"animation_edits":       []any{},
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func studioDBRequalificationLayer(t *testing.T, stateHash string) json.RawMessage {
	t.Helper()
	value := map[string]any{
		"schema_version":                    AuthoringLayerSchema,
		"source_product_state_hash":         stateHash,
		"authority_class":                   "OPTIONAL_USER_AUTHORING_LAYER",
		"can_promote_product_directly":      false,
		"requires_compiler_requalification": true,
		"view_id":                           "V0",
		"metadata": map[string]any{
			"canonical_bundle_mutated": false,
			"animation_track_space":    "ANIMATOR_CONTROLS",
			"court_marker":             "animation-requalification",
		},
		"rig_pose_edits":        []any{},
		"mesh_vertex_edits":     []any{},
		"mesh_topology_edits":   []any{},
		"weight_edits":          []any{},
		"binding_edits":         []any{},
		"presentation_override": nil,
		"rig_topology_edits":    []any{},
		"ik_constraints":        []any{},
		"animation_edits": []any{
			map[string]any{
				"clip_id":          "USER_DB_COURT",
				"display_name":     "DB Court",
				"duration_seconds": 1.0,
				"fps":              30.0,
				"loop":             true,
				"tracks": []any{
					map[string]any{
						"control_id": "C0",
						"keys": []any{
							map[string]any{
								"time_seconds":     0.0,
								"rotation_degrees": 0.0,
								"translation_xy":   []any{0.0, 0.0},
								"scale_xy":         []any{1.0, 1.0},
								"curve":            "linear",
							},
							map[string]any{
								"time_seconds":     1.0,
								"rotation_degrees": 15.0,
								"translation_xy":   []any{1.0, 0.0},
								"scale_xy":         []any{1.0, 1.0},
								"curve":            "linear",
							},
						},
					},
				},
			},
		},
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
