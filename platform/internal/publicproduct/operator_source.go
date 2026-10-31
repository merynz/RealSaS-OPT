package publicproduct

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	subjectinput "github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

// StartFromSourceObservationBytes is an operator-only ingress for already
// materialized SourceObservation.v1. It deliberately bypasses raw public image
// intake, but not product.input or any downstream product capability. The
// caller is responsible for producing codec-valid bytes from exact observed
// evidence; the normal compiler/proof/ProductRevision path remains unchanged.
func StartFromSourceObservationBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	codeSHA string,
	payload []byte,
	displayName string,
	actor string,
) (StartResponse, error) {
	if pool == nil || store == nil || strings.TrimSpace(actor) == "" {
		return StartResponse{}, errors.New("pool, artifact store and actor are required")
	}
	if len(payload) == 0 || len(payload) > MaxSourceObservationBytes {
		return StartResponse{}, errors.New("PUBLIC_PRODUCT_SOURCE_OBSERVATION_SIZE_INVALID")
	}
	object, err := store.PutBytes(ctx, payload)
	if err != nil {
		return StartResponse{}, err
	}
	registered, err := registry.Import(ctx, pool, store, registry.ImportRequest{
		ArtifactType:  "RealSaS.SourceObservation",
		SchemaVersion: "v1",
		Object:        object,
		SourceURI:     "operator-exact-source-observation://" + object.ContentSHA256,
		CreatedBy:     actor,
	})
	if err != nil {
		return StartResponse{}, err
	}
	return StartFromSourceObservation(
		ctx, pool, codeSHA, registered.ArtifactID, displayName, actor,
	)
}

// StartFromSourceObservation creates a public compile identity directly from a
// registered immutable SourceObservation artifact. This is intentionally not a
// browser/public upload API. It exists so controlled exact-8 research/product
// inputs can exercise the same shipping capability DAG while front/back input
// normalization remains a separate replaceable product.input concern.
func StartFromSourceObservation(
	ctx context.Context,
	pool *pgxpool.Pool,
	codeSHA string,
	sourceObservationID uuid.UUID,
	displayName string,
	actor string,
) (StartResponse, error) {
	if pool == nil || sourceObservationID == uuid.Nil || strings.TrimSpace(actor) == "" {
		return StartResponse{}, errors.New("pool, source_observation_id and actor are required")
	}
	var artifactType, schemaVersion string
	if err := pool.QueryRow(ctx, `
		SELECT t.name,t.schema_version
		FROM artifacts a JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE a.id=$1
	`, sourceObservationID).Scan(&artifactType, &schemaVersion); err != nil {
		return StartResponse{}, err
	}
	if artifactType != "RealSaS.SourceObservation" || schemaVersion != "v1" {
		return StartResponse{}, errors.New("PUBLIC_PRODUCT_QUALIFIED_SOURCE_OBSERVATION_REQUIRED")
	}

	channel, err := ActiveProductionRelease(ctx, pool)
	if err != nil {
		return StartResponse{}, err
	}
	if err := release.RequireCanonicalMain(ctx, pool, channel.EngineReleaseID, codeSHA); err != nil {
		return StartResponse{}, err
	}

	publicID := "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	displayName = strings.TrimSpace(displayName)
	if displayName == "" {
		displayName = "Exact 8-View Character"
	}
	if len(displayName) > 300 {
		return StartResponse{}, errors.New("display_name exceeds 300 characters")
	}
	subject, err := subjectinput.EnsureSubject(ctx, pool, subjectinput.SubjectRequest{
		Slug:        "compile-" + strings.TrimPrefix(publicID, "cmp_"),
		DisplayName: displayName,
		CreatedBy:   actor,
	})
	if err != nil {
		return StartResponse{}, err
	}

	receipt, err := command.SubmitProductCapabilityCompile(ctx, pool, command.ProductCapabilityCompileRequest{
		EngineReleaseID:  channel.EngineReleaseID,
		SubjectID:        subject.SubjectID,
		InputArtifactIDs: []uuid.UUID{sourceObservationID},
		Parameters: map[string]any{
			"public_compile_id": publicID,
			"source_authority":  "QUALIFIED_SOURCE_OBSERVATION_V1",
			"ingress":           "OPERATOR_EXACT_SOURCE_OBSERVATION",
		},
		IdempotencyKey: "operator-product:" + publicID + ":compile",
		RequestedBy:    actor,
	})
	if err != nil {
		return StartResponse{}, err
	}

	sessionID := uuid.New()
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
			INSERT INTO public_product_compile_sessions(
				id,public_id,subject_id,engine_release_id,release_channel_lock_version,
				source_observation_artifact_id,attempt_id,state,current_capability
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'QUEUED','product.input')
		`, sessionID, publicID, subject.SubjectID, channel.EngineReleaseID, channel.LockVersion,
			sourceObservationID, receipt.AttemptID); err != nil {
			return err
		}
		payload, _ := json.Marshal(map[string]any{
			"public_id":                      publicID,
			"attempt_id":                     receipt.AttemptID,
			"source_observation_artifact_id": sourceObservationID,
			"release_channel_lock_version":   channel.LockVersion,
			"ingress":                        "OPERATOR_EXACT_SOURCE_OBSERVATION",
			"raw_public_views_created":       false,
			"intake_job_created":             false,
		})
		_, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,attempt_id,artifact_id,payload)
			VALUES ($1,'OPERATOR_PRODUCT_EXACT_SOURCE_OBSERVATION_ACCEPTED',$2,$3,$4,$5)
		`, actor, subject.SubjectID, receipt.AttemptID, sourceObservationID, payload)
		return err
	})
	if err != nil {
		return StartResponse{}, err
	}
	return StartResponse{
		CompileID: publicID,
		Status:    "QUEUED",
		Stage:     "INPUT",
		Detail:    "Exact SourceObservation accepted. Product compile queued.",
		Terminal:  false,
	}, nil
}
