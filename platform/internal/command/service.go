package command

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

const (
	CompileSubject = "COMPILE_SUBJECT"
	RenderProduct  = "RENDER_PRODUCT"
)

var (
	ErrIdempotencyConflict       = errors.New("idempotency key command identity conflict")
	ErrSubjectNotFound           = errors.New("command subject not found")
	ErrSubjectInputMismatch      = errors.New("compile subject input mismatch")
	ErrProductReleaseRequired    = errors.New("compile requires PRODUCT engine release")
	ErrRevisionSubjectMismatch   = errors.New("render product revision subject mismatch")
	ErrMotionArtifactNotFound    = errors.New("render motion artifact not found")
	ErrMotionArtifactTypeInvalid = errors.New("render motion artifact must belong to motion domain")
)

type Receipt struct {
	CommandID            uuid.UUID
	CommandType          string
	SubjectID            uuid.UUID
	ReusedIdempotencyKey bool
	AttemptID            *uuid.UUID
	RenderRequestID      *uuid.UUID
}

type CompileRequest struct {
	SubjectID       uuid.UUID
	EngineReleaseID uuid.UUID
	SubjectInputID  uuid.UUID
	TargetStageID   string
	IdempotencyKey  string
	RequestedBy     string
}

type RenderRequest struct {
	SubjectID         uuid.UUID
	ProductRevisionID uuid.UUID
	MotionArtifactID  uuid.UUID
	ViewSpec          map[string]any
	RenderSettings    map[string]any
	IdempotencyKey    string
	RequestedBy       string
}

type commandRow struct {
	ID          uuid.UUID
	CommandType string
	SubjectID   uuid.UUID
	Payload     map[string]any
}

func SubmitCompile(ctx context.Context, pool *pgxpool.Pool, graph *stagegraph.Graph, req CompileRequest) (Receipt, error) {
	if req.SubjectID == uuid.Nil || req.EngineReleaseID == uuid.Nil || req.SubjectInputID == uuid.Nil {
		return Receipt{}, errors.New("compile subject/release/input ids are required")
	}
	if req.TargetStageID == "" {
		productPassStageID, ok := graph.ProductPassStageID()
		if !ok {
			return Receipt{}, errors.New("compile graph has no product-pass authority")
		}
		req.TargetStageID = productPassStageID
	}
	if _, ok := graph.Get(req.TargetStageID); !ok {
		return Receipt{}, fmt.Errorf("unknown target stage %s", req.TargetStageID)
	}
	if req.IdempotencyKey == "" || req.RequestedBy == "" {
		return Receipt{}, errors.New("idempotency_key and requested_by are required")
	}

	var out Receipt
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		existing, found, err := findCommandByIdempotency(ctx, tx, req.IdempotencyKey)
		if err != nil {
			return err
		}
		if found {
			if existing.CommandType != CompileSubject || existing.SubjectID != req.SubjectID ||
				payloadString(existing.Payload, "engine_release_id") != req.EngineReleaseID.String() ||
				payloadString(existing.Payload, "subject_input_id") != req.SubjectInputID.String() ||
				payloadString(existing.Payload, "target_stage_id") != req.TargetStageID {
				return ErrIdempotencyConflict
			}
			attemptID, err := payloadUUID(existing.Payload, "attempt_id")
			if err != nil {
				return err
			}
			out = Receipt{
				CommandID:            existing.ID,
				CommandType:          existing.CommandType,
				SubjectID:            req.SubjectID,
				ReusedIdempotencyKey: true,
				AttemptID:            &attemptID,
			}
			return nil
		}

		var subject uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", req.SubjectID).Scan(&subject); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrSubjectNotFound
			}
			return err
		}

		var inputSubject uuid.UUID
		var inputManifestSHA string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,manifest_sha256
			FROM subject_inputs
			WHERE id=$1
		`, req.SubjectInputID).Scan(&inputSubject, &inputManifestSHA); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrSubjectInputMismatch
			}
			return err
		}
		if inputSubject != req.SubjectID {
			return ErrSubjectInputMismatch
		}

		var releaseSHA, purpose string
		if err := tx.QueryRow(ctx, `
			SELECT release_sha256,purpose
			FROM engine_releases
			WHERE id=$1
		`, req.EngineReleaseID).Scan(&releaseSHA, &purpose); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return release.ErrReleaseNotFound
			}
			return err
		}
		if release.Purpose(purpose) != release.PurposeProduct {
			return ErrProductReleaseRequired
		}

		spec := map[string]any{
			"schema":                        "RealSaS.CompileSubjectCommandSpec.v1",
			"subject_id":                    req.SubjectID.String(),
			"engine_release_id":             req.EngineReleaseID.String(),
			"engine_release_sha256":         releaseSHA,
			"subject_input_id":              req.SubjectInputID.String(),
			"subject_input_manifest_sha256": inputManifestSHA,
			"target_stage_id":               req.TargetStageID,
		}
		specSHA, err := semantic.JSONSHA256(spec)
		if err != nil {
			return err
		}

		attemptID := uuid.New()
		commandID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempts
			  (id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
			VALUES ($1,$2,$3,'compile_candidate',$4,$5,'OPEN')
		`, attemptID, req.SubjectID, req.EngineReleaseID, specSHA, req.RequestedBy); err != nil {
			return err
		}

		payload := map[string]any{
			"schema":            "RealSaS.CompileSubjectCommand.v1",
			"command_id":        commandID.String(),
			"attempt_id":        attemptID.String(),
			"subject_id":        req.SubjectID.String(),
			"engine_release_id": req.EngineReleaseID.String(),
			"subject_input_id":  req.SubjectInputID.String(),
			"target_stage_id":   req.TargetStageID,
		}
		if err := insertCommand(ctx, tx, commandID, CompileSubject, req.SubjectID, req.IdempotencyKey, payload); err != nil {
			return err
		}
		if err := appendOutbox(ctx, tx, "Command", commandID, "CompileSubjectRequested", map[string]any{"command_id": commandID.String()}); err != nil {
			return err
		}
		if err := appendAudit(ctx, tx, req.RequestedBy, "COMPILE_COMMAND_ACCEPTED", req.SubjectID, &attemptID, nil, map[string]any{
			"command_id":        commandID.String(),
			"engine_release_id": req.EngineReleaseID.String(),
			"subject_input_id":  req.SubjectInputID.String(),
			"target_stage_id":   req.TargetStageID,
		}); err != nil {
			return err
		}

		out = Receipt{
			CommandID:   commandID,
			CommandType: CompileSubject,
			SubjectID:   req.SubjectID,
			AttemptID:   &attemptID,
		}
		return nil
	})
	return out, err
}

func SubmitRender(ctx context.Context, pool *pgxpool.Pool, req RenderRequest) (Receipt, error) {
	if req.SubjectID == uuid.Nil || req.ProductRevisionID == uuid.Nil || req.MotionArtifactID == uuid.Nil {
		return Receipt{}, errors.New("render subject/revision/motion ids are required")
	}
	if req.IdempotencyKey == "" || req.RequestedBy == "" {
		return Receipt{}, errors.New("idempotency_key and requested_by are required")
	}
	if req.ViewSpec == nil {
		req.ViewSpec = map[string]any{}
	}
	if req.RenderSettings == nil {
		req.RenderSettings = map[string]any{}
	}

	var out Receipt
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		existing, found, err := findCommandByIdempotency(ctx, tx, req.IdempotencyKey)
		if err != nil {
			return err
		}
		if found && (existing.CommandType != RenderProduct || existing.SubjectID != req.SubjectID) {
			return ErrIdempotencyConflict
		}

		var revisionSubject uuid.UUID
		var revisionManifestSHA string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,manifest_sha256
			FROM product_revisions
			WHERE id=$1
		`, req.ProductRevisionID).Scan(&revisionSubject, &revisionManifestSHA); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrRevisionSubjectMismatch
			}
			return err
		}
		if revisionSubject != req.SubjectID {
			return ErrRevisionSubjectMismatch
		}

		var motionSemanticSHA, motionDomain string
		if err := tx.QueryRow(ctx, `
			SELECT a.semantic_sha256,t.domain
			FROM artifacts a
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE a.id=$1
		`, req.MotionArtifactID).Scan(&motionSemanticSHA, &motionDomain); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrMotionArtifactNotFound
			}
			return err
		}
		if motionDomain != "motion" {
			return ErrMotionArtifactTypeInvalid
		}

		spec := domain.RenderRequestSpec{
			ProductRevisionManifestSHA256: revisionManifestSHA,
			MotionArtifactSemanticSHA256:  motionSemanticSHA,
			ViewSpec:                      req.ViewSpec,
			RenderSettings:                req.RenderSettings,
		}
		semanticSHA, err := spec.SemanticSHA256()
		if err != nil {
			return err
		}
		if found {
			if payloadString(existing.Payload, "product_revision_id") != req.ProductRevisionID.String() ||
				payloadString(existing.Payload, "render_request_semantic_sha256") != semanticSHA {
				return ErrIdempotencyConflict
			}
			renderID, err := payloadUUID(existing.Payload, "render_request_id")
			if err != nil {
				return err
			}
			out = Receipt{
				CommandID:            existing.ID,
				CommandType:          existing.CommandType,
				SubjectID:            req.SubjectID,
				ReusedIdempotencyKey: true,
				RenderRequestID:      &renderID,
			}
			return nil
		}

		viewJSON, err := json.Marshal(req.ViewSpec)
		if err != nil {
			return err
		}
		settingsJSON, err := json.Marshal(req.RenderSettings)
		if err != nil {
			return err
		}
		renderRequestID := uuid.New()
		if err := tx.QueryRow(ctx, `
			INSERT INTO render_requests
			  (id,subject_id,product_revision_id,motion_artifact_id,view_spec,render_settings,semantic_sha256,idempotency_key)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
			ON CONFLICT (semantic_sha256) DO UPDATE
			SET semantic_sha256=EXCLUDED.semantic_sha256
			RETURNING id
		`, renderRequestID, req.SubjectID, req.ProductRevisionID, req.MotionArtifactID, viewJSON, settingsJSON, semanticSHA, "render:"+semanticSHA).Scan(&renderRequestID); err != nil {
			return err
		}

		commandID := uuid.New()
		payload := map[string]any{
			"schema":                         "RealSaS.RenderProductCommand.v1",
			"command_id":                     commandID.String(),
			"subject_id":                     req.SubjectID.String(),
			"product_revision_id":            req.ProductRevisionID.String(),
			"render_request_id":              renderRequestID.String(),
			"render_request_semantic_sha256": semanticSHA,
		}
		if err := insertCommand(ctx, tx, commandID, RenderProduct, req.SubjectID, req.IdempotencyKey, payload); err != nil {
			return err
		}
		if err := appendOutbox(ctx, tx, "Command", commandID, "RenderProductRequested", map[string]any{"command_id": commandID.String()}); err != nil {
			return err
		}
		if err := appendAudit(ctx, tx, req.RequestedBy, "RENDER_COMMAND_ACCEPTED", req.SubjectID, nil, &req.ProductRevisionID, map[string]any{
			"command_id":                     commandID.String(),
			"render_request_id":              renderRequestID.String(),
			"render_request_semantic_sha256": semanticSHA,
		}); err != nil {
			return err
		}

		out = Receipt{
			CommandID:       commandID,
			CommandType:     RenderProduct,
			SubjectID:       req.SubjectID,
			RenderRequestID: &renderRequestID,
		}
		return nil
	})
	return out, err
}

func findCommandByIdempotency(ctx context.Context, tx pgx.Tx, key string) (commandRow, bool, error) {
	var row commandRow
	var payload []byte
	err := tx.QueryRow(ctx, `
		SELECT id,command_type,subject_id,payload
		FROM commands
		WHERE idempotency_key=$1
	`, key).Scan(&row.ID, &row.CommandType, &row.SubjectID, &payload)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return commandRow{}, false, nil
		}
		return commandRow{}, false, err
	}
	if err := json.Unmarshal(payload, &row.Payload); err != nil {
		return commandRow{}, false, err
	}
	return row, true, nil
}

func insertCommand(ctx context.Context, tx pgx.Tx, id uuid.UUID, commandType string, subjectID uuid.UUID, idempotencyKey string, payload map[string]any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload)
		VALUES ($1,$2,$3,$4,$5)
	`, id, commandType, subjectID, idempotencyKey, body)
	return err
}

func appendOutbox(ctx context.Context, tx pgx.Tx, aggregateType string, aggregateID uuid.UUID, eventType string, payload map[string]any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload)
		VALUES ($1,$2,$3,$4)
	`, aggregateType, aggregateID, eventType, body)
	return err
}

func appendAudit(ctx context.Context, tx pgx.Tx, actor, action string, subjectID uuid.UUID, attemptID, revisionID *uuid.UUID, payload map[string]any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO audit_events(actor,action,subject_id,attempt_id,product_revision_id,payload)
		VALUES ($1,$2,$3,$4,$5,$6)
	`, actor, action, subjectID, attemptID, revisionID, body)
	return err
}

func payloadUUID(payload map[string]any, key string) (uuid.UUID, error) {
	raw, ok := payload[key].(string)
	if !ok || raw == "" {
		return uuid.Nil, fmt.Errorf("command payload missing %s", key)
	}
	return uuid.Parse(raw)
}

func payloadString(payload map[string]any, key string) string {
	raw, _ := payload[key].(string)
	return raw
}
