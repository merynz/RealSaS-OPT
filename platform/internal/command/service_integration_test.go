package command

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func commandGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func commandReleaseManifest(g *stagegraph.Graph, name string) release.Manifest {
	rows := make([]release.StageBinding, 0, 46)
	for _, s := range g.Stages() {
		rows = append(rows, release.StageBinding{
			Ordinal:              s.Ordinal,
			StageID:              s.ID,
			ImplementationSHA256: repeatCommandHex("1"),
			PolicySHA256:         repeatCommandHex("2"),
			SemanticParameters:   map[string]any{"contract_epoch": 1},
		})
	}
	return release.Manifest{Name: name, Purpose: release.PurposeProduct, Stages: rows}
}

func TestCompileCommandIsTransactionalAndIdempotent(t *testing.T) {
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
	g := commandGraph(t)

	subjectID := uuid.New()
	typeID := uuid.New()
	sourceID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "compile-"+subjectID.String(), "Compile Command"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','source')", typeID, "RealSaS.SourceImage."+subjectID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'SOURCE_IMPORT',$6,$7,'{}',$8)
	`, sourceID, typeID, repeatCommandHex("3"), repeatCommandHex("4"), "cas/source/"+sourceID.String(), repeatCommandHex("5"), repeatCommandHex("6"), time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	sealedInput, err := input.Seal(ctx, pool, subjectID, []input.Binding{{Role: "source", ArtifactID: sourceID}}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	sealedRelease, err := release.Seal(ctx, pool, g, commandReleaseManifest(g, "compile-command-"+subjectID.String()), "ci")
	if err != nil {
		t.Fatal(err)
	}

	req := CompileRequest{
		SubjectID:          subjectID,
		EngineReleaseID:    sealedRelease.ReleaseID,
		SubjectInputID:     sealedInput.SubjectInputID,
		CompilerRunID:      "run-" + subjectID.String(),
		RunManifestPath:    "/authority/run_manifest.json",
		RunLedgerPath:      "/authority/ACTIVE_RUN_V2.json",
		PipelinePlanSHA256: repeatCommandHex("d"),
		IdempotencyKey:     "compile:" + subjectID.String(),
		RequestedBy:        "ci",
	}
	first, err := SubmitCompile(ctx, pool, g, req)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SubmitCompile(ctx, pool, g, req)
	if err != nil {
		t.Fatal(err)
	}
	if !second.ReusedIdempotencyKey || first.CommandID != second.CommandID || *first.AttemptID != *second.AttemptID {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	var attempts, events int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM attempts WHERE id=$1", *first.AttemptID).Scan(&attempts); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM outbox_events WHERE aggregate_id=$1 AND event_type='CompileSubjectRequested'", first.CommandID).Scan(&events); err != nil {
		t.Fatal(err)
	}
	if attempts != 1 || events != 1 {
		t.Fatalf("attempts=%d outbox=%d", attempts, events)
	}

	conflict := req
	conflict.TargetStageID = "45_DYNAMIC_VISUAL_INTEGRITY_PROOF"
	if _, err := SubmitCompile(ctx, pool, g, conflict); !errors.Is(err, ErrIdempotencyConflict) {
		t.Fatalf("expected idempotency conflict, got %v", err)
	}
}

func TestRenderCommandsShareExactSemanticRenderRequest(t *testing.T) {
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

	subjectID := uuid.New()
	attemptID := uuid.New()
	revisionID := uuid.New()
	typeID := uuid.New()
	motionID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "render-"+subjectID.String(), "Render Command"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO attempts(id,subject_id,kind,spec_sha256,created_by,final_state) VALUES ($1,$2,'compile_candidate',$3,'ci','QUALIFIED')", attemptID, subjectID, repeatCommandHex("7")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','motion')", typeID, "RealSaS.Motion."+subjectID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'40_MOTION_COMPILE_RUN',$6,$7,'{}',$8)
	`, motionID, typeID, repeatCommandHex("8"), repeatCommandHex("9"), "cas/motion/"+motionID.String(), repeatCommandHex("a"), repeatCommandHex("b"), time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revisions(id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at)
		VALUES ($1,$2,1,$3,$4,now())
	`, revisionID, subjectID, repeatCommandHex("c"), attemptID); err != nil {
		t.Fatal(err)
	}

	base := RenderRequest{
		SubjectID:         subjectID,
		ProductRevisionID: revisionID,
		MotionArtifactID:  motionID,
		ViewSpec:          map[string]any{"view": "front"},
		RenderSettings:    map[string]any{"fps": 24},
		RequestedBy:       "ci",
	}
	firstReq := base
	firstReq.IdempotencyKey = "render-a:" + subjectID.String()
	first, err := SubmitRender(ctx, pool, firstReq)
	if err != nil {
		t.Fatal(err)
	}
	secondReq := base
	secondReq.IdempotencyKey = "render-b:" + subjectID.String()
	second, err := SubmitRender(ctx, pool, secondReq)
	if err != nil {
		t.Fatal(err)
	}
	if *first.RenderRequestID != *second.RenderRequestID || first.CommandID == second.CommandID {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	var renderRows, outboxRows int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM render_requests WHERE id=$1", *first.RenderRequestID).Scan(&renderRows); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM outbox_events WHERE event_type='RenderProductRequested' AND aggregate_id IN ($1,$2)", first.CommandID, second.CommandID).Scan(&outboxRows); err != nil {
		t.Fatal(err)
	}
	if renderRows != 1 || outboxRows != 2 {
		t.Fatalf("render rows=%d outbox=%d", renderRows, outboxRows)
	}
	third, err := SubmitRender(ctx, pool, firstReq)
	if err != nil {
		t.Fatal(err)
	}
	if !third.ReusedIdempotencyKey || third.CommandID != first.CommandID || *third.RenderRequestID != *first.RenderRequestID {
		t.Fatalf("first=%+v third=%+v", first, third)
	}

	conflict := firstReq
	conflict.RenderSettings = map[string]any{"fps": 30}
	if _, err := SubmitRender(ctx, pool, conflict); !errors.Is(err, ErrIdempotencyConflict) {
		t.Fatalf("expected render idempotency conflict, got %v", err)
	}
}

func repeatCommandHex(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
