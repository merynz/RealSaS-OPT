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

func Start(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	codeSHA string,
	req StartRequest,
	actor string,
) (StartResponse, error) {
	if pool == nil || store == nil || strings.TrimSpace(actor) == "" {
		return StartResponse{}, errors.New("pool, artifact store and actor are required")
	}
	views, err := validateViews(req.Views)
	if err != nil {
		return StartResponse{}, err
	}
	channel, err := ActiveProductionRelease(ctx, pool)
	if err != nil {
		return StartResponse{}, err
	}
	if err := release.RequireCanonicalMain(ctx, pool, channel.EngineReleaseID, codeSHA); err != nil {
		return StartResponse{}, err
	}

	publicID := "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	displayName := strings.TrimSpace(req.DisplayName)
	if displayName == "" {
		displayName = "Untitled Character"
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

	type importedView struct {
		Slot       string
		Name       string
		MIMEType   string
		ArtifactID uuid.UUID
	}
	imported := make([]importedView, 0, len(views))
	for _, view := range views {
		object, err := store.PutBytes(ctx, view.BytesBase64)
		if err != nil {
			return StartResponse{}, err
		}
		registered, err := registry.Import(ctx, pool, store, registry.ImportRequest{
			ArtifactType:  "RealSaS.ProductSourceImage",
			SchemaVersion: "v1",
			Object:        object,
			SourceURI:     "public-upload://" + publicID + "/" + view.Slot,
			CreatedBy:     actor,
		})
		if err != nil {
			return StartResponse{}, err
		}
		imported = append(imported, importedView{
			Slot:       view.Slot,
			Name:       view.Name,
			MIMEType:   view.MIMEType,
			ArtifactID: registered.ArtifactID,
		})
	}

	sessionID := uuid.New()
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
			INSERT INTO public_product_compile_sessions(
				id,public_id,subject_id,engine_release_id,release_channel_lock_version,
				state,current_capability
			) VALUES ($1,$2,$3,$4,$5,'WAITING_SOURCE_MATERIALIZATION','product.input')
		`, sessionID, publicID, subject.SubjectID, channel.EngineReleaseID, channel.LockVersion); err != nil {
			return err
		}
		for _, view := range imported {
			if _, err := tx.Exec(ctx, `
				INSERT INTO public_product_compile_views(session_id,slot,artifact_id,mime_type,original_name)
				VALUES ($1,$2,$3,$4,$5)
			`, sessionID, view.Slot, view.ArtifactID, view.MIMEType, view.Name); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO public_product_intake_jobs(session_id,state)
			VALUES ($1,'QUEUED')
		`, sessionID); err != nil {
			return err
		}
		viewSlots := make([]string, 0, len(imported))
		for _, view := range imported {
			viewSlots = append(viewSlots, view.Slot)
		}
		payload, _ := json.Marshal(map[string]any{
			"public_id":                    publicID,
			"view_slots":                   viewSlots,
			"release_channel_lock_version": channel.LockVersion,
			"authority":                    "RAW_EXTERNAL_EVIDENCE_ONLY",
			"intake_job_state":             "QUEUED",
		})
		_, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,payload)
			VALUES ($1,'PUBLIC_PRODUCT_COMPILE_INTAKE_ACCEPTED',$2,$3)
		`, actor, subject.SubjectID, payload)
		return err
	})
	if err != nil {
		return StartResponse{}, err
	}
	return StartResponse{
		CompileID: publicID,
		Status:    "WAITING_SOURCE_MATERIALIZATION",
		Stage:     "INPUT",
		Detail:    "Source accepted. Preparing qualified character observations.",
		Terminal:  false,
	}, nil
}

func AttachSourceObservation(
	ctx context.Context,
	pool *pgxpool.Pool,
	codeSHA string,
	publicID string,
	sourceObservationID uuid.UUID,
	actor string,
) (Status, error) {
	if pool == nil || strings.TrimSpace(publicID) == "" || sourceObservationID == uuid.Nil || strings.TrimSpace(actor) == "" {
		return Status{}, errors.New("pool, public_id, source_observation_id and actor are required")
	}
	session, err := loadSession(ctx, pool, publicID)
	if err != nil {
		return Status{}, err
	}
	if session.SourceObservationID != nil && *session.SourceObservationID != sourceObservationID {
		return Status{}, errors.New("PUBLIC_PRODUCT_SOURCE_OBSERVATION_IMMUTABLE")
	}
	var artifactType, schemaVersion string
	if err := pool.QueryRow(ctx, `
		SELECT t.name,t.schema_version
		FROM artifacts a JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE a.id=$1
	`, sourceObservationID).Scan(&artifactType, &schemaVersion); err != nil {
		return Status{}, err
	}
	if artifactType != "RealSaS.SourceObservation" || schemaVersion != "v1" {
		return Status{}, errors.New("PUBLIC_PRODUCT_QUALIFIED_SOURCE_OBSERVATION_REQUIRED")
	}
	if err := release.RequireCanonicalMain(ctx, pool, session.EngineReleaseID, codeSHA); err != nil {
		return Status{}, err
	}

	receipt, err := command.SubmitProductCapabilityCompile(ctx, pool, command.ProductCapabilityCompileRequest{
		EngineReleaseID:  session.EngineReleaseID,
		SubjectID:        session.SubjectID,
		InputArtifactIDs: []uuid.UUID{sourceObservationID},
		Parameters: map[string]any{
			"public_compile_id": publicID,
			"source_authority":  "QUALIFIED_SOURCE_OBSERVATION_V1",
		},
		IdempotencyKey: "public-product:" + publicID + ":compile",
		RequestedBy:    actor,
	})
	if err != nil {
		return Status{}, err
	}

	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var existingSource, existingAttempt *uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT source_observation_artifact_id,attempt_id
			FROM public_product_compile_sessions
			WHERE id=$1 FOR UPDATE
		`, session.ID).Scan(&existingSource, &existingAttempt); err != nil {
			return err
		}
		if existingSource != nil && *existingSource != sourceObservationID {
			return errors.New("PUBLIC_PRODUCT_SOURCE_OBSERVATION_IMMUTABLE")
		}
		if existingAttempt != nil && *existingAttempt != receipt.AttemptID {
			return errors.New("PUBLIC_PRODUCT_ATTEMPT_IDENTITY_DRIFT")
		}
		if _, err := tx.Exec(ctx, `
			UPDATE public_product_compile_sessions
			SET source_observation_artifact_id=$2,attempt_id=$3,state='QUEUED',
				current_capability='product.input',error_code=NULL
			WHERE id=$1
		`, session.ID, sourceObservationID, receipt.AttemptID); err != nil {
			return err
		}
		payload, _ := json.Marshal(map[string]any{
			"public_id":                      publicID,
			"attempt_id":                     receipt.AttemptID,
			"source_observation_artifact_id": sourceObservationID,
		})
		_, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,attempt_id,artifact_id,payload)
			VALUES ($1,'PUBLIC_PRODUCT_COMPILE_QUEUED',$2,$3,$4,$5)
		`, actor, session.SubjectID, receipt.AttemptID, sourceObservationID, payload)
		return err
	})
	if err != nil {
		return Status{}, err
	}
	return GetStatus(ctx, pool, publicID)
}
