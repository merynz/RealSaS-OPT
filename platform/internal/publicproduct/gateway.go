package publicproduct

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

const (
	ProductionChannel = "PRODUCTION"
	maxPublicViewBytes = 6 << 20
)

var (
	ErrReleaseChannelUnbound = errors.New("PUBLIC_PRODUCT_RELEASE_CHANNEL_UNBOUND")
	ErrCompileNotFound       = errors.New("PUBLIC_PRODUCT_COMPILE_NOT_FOUND")
)

type ViewUpload struct {
	Slot        string `json:"slot"`
	Name        string `json:"name"`
	MIMEType    string `json:"mime_type"`
	BytesBase64 []byte `json:"bytes_base64"`
}

type StartRequest struct {
	DisplayName string       `json:"display_name,omitempty"`
	Views       []ViewUpload `json:"views"`
}

type StartResponse struct {
	CompileID string `json:"compile_id"`
	Status    string `json:"status"`
	Stage     string `json:"stage"`
	Detail    string `json:"detail"`
	Terminal  bool   `json:"terminal"`
}

type Status struct {
	CompileID   string `json:"compile_id"`
	Status      string `json:"status"`
	Stage       string `json:"stage"`
	Detail      string `json:"detail"`
	Terminal    bool   `json:"terminal"`
	ResultReady bool   `json:"result_ready"`
}

type ReleaseChannel struct {
	Channel         string    `json:"channel"`
	EngineReleaseID uuid.UUID `json:"engine_release_id"`
	LockVersion     int64     `json:"lock_version"`
}

type sessionRecord struct {
	ID                  uuid.UUID
	PublicID            string
	SubjectID           uuid.UUID
	EngineReleaseID     uuid.UUID
	ChannelLockVersion  int64
	SourceObservationID *uuid.UUID
	AttemptID           *uuid.UUID
	ProductRevisionID   *uuid.UUID
	State               string
	CurrentCapability   *string
	ErrorCode            *string
}

func BindProductionRelease(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	codeSHA string,
	actor string,
) (ReleaseChannel, error) {
	if pool == nil || releaseID == uuid.Nil || strings.TrimSpace(codeSHA) == "" || strings.TrimSpace(actor) == "" {
		return ReleaseChannel{}, errors.New("pool, release_id, code_sha and actor are required")
	}
	if err := release.RequireCanonicalMain(ctx, pool, releaseID, codeSHA); err != nil {
		return ReleaseChannel{}, err
	}
	var purpose string
	if err := pool.QueryRow(ctx, `SELECT purpose FROM engine_releases WHERE id=$1`, releaseID).Scan(&purpose); err != nil {
		return ReleaseChannel{}, err
	}
	if purpose != string(release.PurposeProduct) {
		return ReleaseChannel{}, errors.New("PUBLIC_PRODUCT_RELEASE_MUST_BE_PRODUCT")
	}

	var out ReleaseChannel
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
			INSERT INTO product_release_channels(channel,engine_release_id,lock_version,updated_by,updated_at)
			VALUES ('PRODUCTION',$1,0,$2,now())
			ON CONFLICT (channel) DO UPDATE SET
				engine_release_id=EXCLUDED.engine_release_id,
				lock_version=product_release_channels.lock_version+1,
				updated_by=EXCLUDED.updated_by,
				updated_at=now()
		`, releaseID, actor); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, `
			SELECT channel,engine_release_id,lock_version
			FROM product_release_channels WHERE channel='PRODUCTION'
		`).Scan(&out.Channel, &out.EngineReleaseID, &out.LockVersion); err != nil {
			return err
		}
		payload, _ := json.Marshal(map[string]any{
			"channel": ProductionChannel, "engine_release_id": releaseID, "lock_version": out.LockVersion,
		})
		_, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,payload)
			VALUES ($1,'PUBLIC_PRODUCT_RELEASE_CHANNEL_BOUND',$2)
		`, actor, payload)
		return err
	})
	return out, err
}

func ActiveProductionRelease(ctx context.Context, pool *pgxpool.Pool) (ReleaseChannel, error) {
	if pool == nil {
		return ReleaseChannel{}, errors.New("pool is required")
	}
	var out ReleaseChannel
	if err := pool.QueryRow(ctx, `
		SELECT channel,engine_release_id,lock_version
		FROM product_release_channels
		WHERE channel='PRODUCTION'
	`).Scan(&out.Channel, &out.EngineReleaseID, &out.LockVersion); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return ReleaseChannel{}, ErrReleaseChannelUnbound
		}
		return ReleaseChannel{}, err
	}
	return out, nil
}

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
			Slot: view.Slot, Name: view.Name, MIMEType: view.MIMEType, ArtifactID: registered.ArtifactID,
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
		payload, _ := json.Marshal(map[string]any{
			"public_id": publicID,
			"view_slots": func() []string {
				out := make([]string, 0, len(imported))
				for _, view := range imported {
					out = append(out, view.Slot)
				}
				return out
			}(),
			"release_channel_lock_version": channel.LockVersion,
			"authority": "RAW_EXTERNAL_EVIDENCE_ONLY",
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
			"public_id": publicID,
			"attempt_id": receipt.AttemptID,
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

func GetStatus(ctx context.Context, pool *pgxpool.Pool, publicID string) (Status, error) {
	session, err := loadSession(ctx, pool, publicID)
	if err != nil {
		return Status{}, err
	}
	if session.AttemptID == nil {
		return Status{
			CompileID: publicID, Status: "WAITING_SOURCE_MATERIALIZATION", Stage: "INPUT",
			Detail: "Source accepted. Preparing qualified character observations.", Terminal: false,
		}, nil
	}

	var finalState string
	if err := pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, *session.AttemptID).Scan(&finalState); err != nil {
		return Status{}, err
	}
	if finalState == "QUALIFIED" {
		var revisionID uuid.UUID
		if err := pool.QueryRow(ctx, `
			SELECT id FROM product_revisions WHERE created_from_attempt_id=$1
		`, *session.AttemptID).Scan(&revisionID); err != nil {
			return Status{}, err
		}
		_, err := pool.Exec(ctx, `
			UPDATE public_product_compile_sessions
			SET state='PASS',current_capability='product.proof',product_revision_id=$2,error_code=NULL
			WHERE id=$1
		`, session.ID, revisionID)
		if err != nil {
			return Status{}, err
		}
		return Status{
			CompileID: publicID, Status: "PASS", Stage: "PROOF",
			Detail: "Product proof passed. Your character is ready.", Terminal: true, ResultReady: true,
		}, nil
	}
	if finalState == "FAILED" {
		_ = markFailed(ctx, pool, session.ID, "PRODUCT_COMPILE_FAILED")
		return Status{
			CompileID: publicID, Status: "FAILED", Stage: publicStage(session.CurrentCapability),
			Detail: "Compile stopped because a product qualification gate failed.", Terminal: true,
		}, nil
	}

	var stageContract, executionStatus string
	var errorCode *string
	err = pool.QueryRow(ctx, `
		SELECT stage_contract,status,error_code
		FROM executions
		WHERE attempt_id=$1
		ORDER BY started_at DESC NULLS LAST, finished_at DESC NULLS LAST
		LIMIT 1
	`, *session.AttemptID).Scan(&stageContract, &executionStatus, &errorCode)
	if errors.Is(err, pgx.ErrNoRows) {
		return Status{
			CompileID: publicID, Status: "QUEUED", Stage: "INPUT",
			Detail: "Compile admitted. Waiting for the first product capability.", Terminal: false,
		}, nil
	}
	if err != nil {
		return Status{}, err
	}
	stage := publicStage(&stageContract)
	if executionStatus == "FAILED" || executionStatus == "ERROR" {
		code := "PRODUCT_CAPABILITY_FAILED"
		if errorCode != nil && strings.TrimSpace(*errorCode) != "" {
			code = *errorCode
		}
		_ = markFailed(ctx, pool, session.ID, code)
		return Status{
			CompileID: publicID, Status: "FAILED", Stage: stage,
			Detail: "Compile stopped because a product capability failed.", Terminal: true,
		}, nil
	}
	_, _ = pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='RUNNING',current_capability=$2,error_code=NULL
		WHERE id=$1
	`, session.ID, stageContract)
	return Status{
		CompileID: publicID, Status: "RUNNING", Stage: stage,
		Detail: "Compiling qualified product state.", Terminal: false,
	}, nil
}

func RawViewArtifactIDs(ctx context.Context, pool *pgxpool.Pool, publicID string) (map[string]uuid.UUID, error) {
	session, err := loadSession(ctx, pool, publicID)
	if err != nil {
		return nil, err
	}
	rows, err := pool.Query(ctx, `
		SELECT slot,artifact_id
		FROM public_product_compile_views
		WHERE session_id=$1 ORDER BY slot
	`, session.ID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]uuid.UUID{}
	for rows.Next() {
		var slot string
		var artifactID uuid.UUID
		if err := rows.Scan(&slot, &artifactID); err != nil {
			return nil, err
		}
		out[slot] = artifactID
	}
	return out, rows.Err()
}

func loadSession(ctx context.Context, pool *pgxpool.Pool, publicID string) (sessionRecord, error) {
	publicID = strings.TrimSpace(publicID)
	if pool == nil || publicID == "" || len(publicID) > 96 {
		return sessionRecord{}, ErrCompileNotFound
	}
	var out sessionRecord
	err := pool.QueryRow(ctx, `
		SELECT id,public_id,subject_id,engine_release_id,release_channel_lock_version,
		       source_observation_artifact_id,attempt_id,product_revision_id,state,
		       current_capability,error_code
		FROM public_product_compile_sessions
		WHERE public_id=$1
	`, publicID).Scan(
		&out.ID, &out.PublicID, &out.SubjectID, &out.EngineReleaseID, &out.ChannelLockVersion,
		&out.SourceObservationID, &out.AttemptID, &out.ProductRevisionID, &out.State,
		&out.CurrentCapability, &out.ErrorCode,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return sessionRecord{}, ErrCompileNotFound
	}
	return out, err
}

func markFailed(ctx context.Context, pool *pgxpool.Pool, sessionID uuid.UUID, code string) error {
	_, err := pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='FAILED',error_code=$2
		WHERE id=$1
	`, sessionID, code)
	return err
}

func publicStage(capability *string) string {
	if capability == nil {
		return "INPUT"
	}
	value := strings.TrimSpace(*capability)
	value = strings.TrimPrefix(value, "product.")
	if value == "" {
		return "INPUT"
	}
	return strings.ToUpper(value)
}

func validateViews(input []ViewUpload) ([]ViewUpload, error) {
	if len(input) < 1 || len(input) > 2 {
		return nil, errors.New("PUBLIC_PRODUCT_REQUIRES_ONE_OR_TWO_SOURCE_VIEWS")
	}
	seen := map[string]struct{}{}
	front := false
	out := make([]ViewUpload, 0, len(input))
	for _, view := range input {
		view.Slot = strings.ToLower(strings.TrimSpace(view.Slot))
		view.MIMEType = strings.ToLower(strings.TrimSpace(view.MIMEType))
		view.Name = strings.TrimSpace(view.Name)
		if view.Slot != "front" && view.Slot != "back" {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SLOT_INVALID")
		}
		if _, duplicate := seen[view.Slot]; duplicate {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SLOT_DUPLICATE")
		}
		seen[view.Slot] = struct{}{}
		if view.Slot == "front" {
			front = true
		}
		if view.Name == "" || len(view.Name) > 300 {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_NAME_INVALID")
		}
		if len(view.BytesBase64) == 0 || len(view.BytesBase64) > maxPublicViewBytes {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SIZE_INVALID")
		}
		if !validImageBytes(view.MIMEType, view.BytesBase64) {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_FORMAT_INVALID")
		}
		out = append(out, view)
	}
	if !front {
		return nil, errors.New("PUBLIC_PRODUCT_FRONT_VIEW_REQUIRED")
	}
	return out, nil
}

func validImageBytes(mimeType string, data []byte) bool {
	switch mimeType {
	case "image/png":
		return len(data) >= 8 && bytes.Equal(data[:8], []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	case "image/jpeg":
		return len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff
	case "image/webp":
		return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
	default:
		return false
	}
}

func debugSession(value sessionRecord) string {
	return fmt.Sprintf("public=%s state=%s", value.PublicID, value.State)
}
