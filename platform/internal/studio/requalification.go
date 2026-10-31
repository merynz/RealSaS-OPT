package studio

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/publicproduct"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
)

var (
	ErrRequalificationNotFound     = errors.New("STUDIO_REQUALIFICATION_NOT_FOUND")
	ErrRequalificationUnsupported  = errors.New("STUDIO_REQUALIFICATION_EDIT_SET_UNSUPPORTED")
	ErrRequalificationHeadRequired = errors.New("STUDIO_REQUALIFICATION_CURRENT_SAVED_HEAD_REQUIRED")
)

type RequalificationRequest struct {
	ExpectedLockVersion int64  `json:"expected_lock_version"`
	RevisionNumber      int64  `json:"revision_number"`
	CreatedBy           string `json:"created_by,omitempty"`
}

type RequalificationStatus struct {
	RequalificationID  string    `json:"requalification_id"`
	ProjectID          string    `json:"project_id"`
	RevisionNumber     int64     `json:"revision_number"`
	State              string    `json:"state"`
	Stage              string    `json:"stage"`
	Detail             string    `json:"detail"`
	InvalidatedDomains []string  `json:"invalidated_domains"`
	ResultCompileID    string    `json:"result_compile_id,omitempty"`
	Terminal           bool      `json:"terminal"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"updated_at"`
}

type requalificationHead struct {
	ProjectUUID     uuid.UUID
	BaseSessionID   uuid.UUID
	BaseRevisionID  uuid.UUID
	BaseManifestSHA string
	RevisionUUID    uuid.UUID
	RevisionNumber  int64
	LockVersion     int64
	PayloadSHA      string
	Payload         json.RawMessage
}

var animationRequalificationSatisfied = []string{
	capability.ProductInputCapabilityID,
	capability.ProductShapeCapabilityID,
	capability.ProductMeshCapabilityID,
	capability.ProductRigCapabilityID,
	capability.ProductSkinCapabilityID,
	capability.ProductMechanicsCapabilityID,
	capability.ProductGraphicsCapabilityID,
}

func RequestRequalification(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	projectID string,
	req RequalificationRequest,
) (RequalificationStatus, error) {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || store == nil || !projectIDPattern.MatchString(projectID) {
		return RequalificationStatus{}, ErrProjectNotFound
	}
	if req.ExpectedLockVersion < 0 || req.RevisionNumber <= 0 {
		return RequalificationStatus{}, ErrRequalificationHeadRequired
	}
	actor := normalizedActor(req.CreatedBy)

	head, err := loadRequalificationHead(ctx, pool, projectID)
	if err != nil {
		return RequalificationStatus{}, err
	}
	if head.LockVersion != req.ExpectedLockVersion || head.RevisionNumber != req.RevisionNumber {
		return RequalificationStatus{}, ErrRequalificationHeadRequired
	}
	if existing, found, err := findRequalificationByRevision(ctx, pool, head.ProjectUUID, head.RevisionUUID); err != nil {
		return RequalificationStatus{}, err
	} else if found {
		return syncRequalification(ctx, pool, projectID, existing)
	}

	canonical, payloadSHA, err := canonicalAuthoringPayload(head.Payload)
	if err != nil {
		return RequalificationStatus{}, err
	}
	if payloadSHA != head.PayloadSHA {
		return RequalificationStatus{}, errors.New("STUDIO_REQUALIFICATION_AUTHORING_PAYLOAD_HASH_DRIFT")
	}
	if err := validateAnimationOnlyRequalification(canonical); err != nil {
		return RequalificationStatus{}, err
	}

	captured, err := registry.CaptureBytes(ctx, pool, store, registry.CaptureRequest{
		ArtifactType:     "RealSaS.StudioAuthoringRevision",
		SchemaVersion:    "v1",
		Payload:          canonical,
		ProducerContract: "RealSaS.StudioAuthoringCapture.v1",
		AuthorityClass:   "STUDIO_AUTHORING_INPUT",
		SemanticParameters: map[string]any{
			"project_id":                        projectID,
			"revision_number":                   head.RevisionNumber,
			"authoring_payload_sha256":          head.PayloadSHA,
			"base_product_revision_id":          head.BaseRevisionID.String(),
			"base_product_manifest_sha256":      head.BaseManifestSHA,
			"supported_requalification_slice":   "ANIMATION_LINEAR_V1",
			"can_promote_product_directly":      false,
			"requires_compiler_requalification": true,
		},
		CreatedBy: actor,
	})
	if err != nil {
		return RequalificationStatus{}, err
	}

	receipt, err := command.SubmitProductRequalification(ctx, pool, command.ProductRequalificationRequest{
		BaseProductRevisionID: head.BaseRevisionID,
		AuthoringArtifactID:   captured.ArtifactID,
		SatisfiedCapabilities: append([]string(nil), animationRequalificationSatisfied...),
		InvalidatedDomains:    []string{"MOTION", "PROOF"},
		IdempotencyKey:        "studio-requal:" + head.ProjectUUID.String() + ":" + head.RevisionUUID.String(),
		RequestedBy:           actor,
	})
	if err != nil {
		return RequalificationStatus{}, err
	}

	requestUUID := uuid.New()
	publicID := "srq_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	invalidatedJSON, _ := json.Marshal([]string{"MOTION", "PROOF"})
	satisfiedJSON, _ := json.Marshal(animationRequalificationSatisfied)
	_, err = pool.Exec(ctx, `
		INSERT INTO studio_requalification_requests(
			id,public_id,project_id,authoring_revision_id,base_product_revision_id,
			authoring_artifact_id,attempt_id,invalidated_domains,satisfied_capability_ids,
			state,created_by
		) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'QUEUED',$10)
		ON CONFLICT (project_id,authoring_revision_id) DO NOTHING
	`, requestUUID, publicID, head.ProjectUUID, head.RevisionUUID, head.BaseRevisionID,
		captured.ArtifactID, receipt.AttemptID, invalidatedJSON, satisfiedJSON, actor)
	if err != nil {
		return RequalificationStatus{}, err
	}
	existing, found, err := findRequalificationByRevision(ctx, pool, head.ProjectUUID, head.RevisionUUID)
	if err != nil {
		return RequalificationStatus{}, err
	}
	if !found {
		return RequalificationStatus{}, errors.New("STUDIO_REQUALIFICATION_REQUEST_INSERT_LOST")
	}
	if existing.AttemptID != receipt.AttemptID || existing.AuthoringArtifactID != captured.ArtifactID {
		return RequalificationStatus{}, errors.New("STUDIO_REQUALIFICATION_REQUEST_IDENTITY_DRIFT")
	}
	return syncRequalification(ctx, pool, projectID, existing)
}

type requalificationRecord struct {
	ID                  uuid.UUID
	PublicID            string
	ProjectUUID         uuid.UUID
	RevisionUUID        uuid.UUID
	RevisionNumber      int64
	BaseSessionID       uuid.UUID
	BaseRevisionID      uuid.UUID
	AuthoringArtifactID uuid.UUID
	AttemptID           uuid.UUID
	State               string
	InvalidatedDomains  []string
	ResultRevisionID    *uuid.UUID
	ResultSessionID     *uuid.UUID
	FailureCode         *string
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

func GetRequalification(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID, requalificationID string,
) (RequalificationStatus, error) {
	projectID = strings.TrimSpace(projectID)
	requalificationID = strings.TrimSpace(requalificationID)
	if pool == nil || !projectIDPattern.MatchString(projectID) || !strings.HasPrefix(requalificationID, "srq_") {
		return RequalificationStatus{}, ErrRequalificationNotFound
	}
	record, err := loadRequalification(ctx, pool, projectID, requalificationID)
	if err != nil {
		return RequalificationStatus{}, err
	}
	return syncRequalification(ctx, pool, projectID, record)
}

func loadRequalificationHead(ctx context.Context, pool *pgxpool.Pool, projectID string) (requalificationHead, error) {
	var out requalificationHead
	var payload []byte
	err := pool.QueryRow(ctx, `
		SELECT p.id,p.source_compile_session_id,p.product_revision_id,pr.manifest_sha256,
		       ar.id,ar.revision_number,h.lock_version,ar.payload_sha256,ar.authoring_payload
		FROM studio_projects p
		JOIN product_revisions pr ON pr.id=p.product_revision_id
		JOIN studio_project_heads h ON h.project_id=p.id
		JOIN studio_authoring_revisions ar ON ar.id=h.authoring_revision_id
		WHERE p.public_id=$1 AND p.state='ACTIVE'
	`, projectID).Scan(
		&out.ProjectUUID, &out.BaseSessionID, &out.BaseRevisionID, &out.BaseManifestSHA,
		&out.RevisionUUID, &out.RevisionNumber, &out.LockVersion, &out.PayloadSHA, &payload,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		if _, projectErr := GetProject(ctx, pool, projectID); projectErr != nil {
			return requalificationHead{}, projectErr
		}
		return requalificationHead{}, ErrRequalificationHeadRequired
	}
	if err != nil {
		return requalificationHead{}, err
	}
	out.Payload = json.RawMessage(payload)
	return out, nil
}

func validateAnimationOnlyRequalification(canonical []byte) error {
	var payload map[string]any
	if err := json.Unmarshal(canonical, &payload); err != nil {
		return ErrAuthoringPayloadInvalid
	}
	for _, field := range []string{
		"rig_pose_edits", "mesh_vertex_edits", "mesh_topology_edits", "weight_edits",
		"binding_edits", "rig_topology_edits", "ik_constraints",
	} {
		if rows, ok := payload[field].([]any); ok && len(rows) > 0 {
			return fmt.Errorf("%w:%s", ErrRequalificationUnsupported, field)
		}
	}
	if presentation, ok := payload["presentation_override"].(map[string]any); ok && presentation != nil {
		if departed, _ := presentation["graphics_evidence_departure"].(bool); departed {
			return fmt.Errorf("%w:presentation_override", ErrRequalificationUnsupported)
		}
		if hidden, ok := presentation["hidden_mesh_ids"].([]any); ok && len(hidden) > 0 {
			return fmt.Errorf("%w:presentation_override", ErrRequalificationUnsupported)
		}
	}
	metadata, _ := payload["metadata"].(map[string]any)
	if str, _ := metadata["animation_track_space"].(string); str != "ANIMATOR_CONTROLS" {
		return fmt.Errorf("%w:animation_track_space", ErrRequalificationUnsupported)
	}
	clips, ok := payload["animation_edits"].([]any)
	if !ok || len(clips) != 1 {
		return fmt.Errorf("%w:animation_clip_cardinality", ErrRequalificationUnsupported)
	}
	clip, ok := clips[0].(map[string]any)
	if !ok {
		return ErrAuthoringPayloadInvalid
	}
	tracks, ok := clip["tracks"].([]any)
	if !ok || len(tracks) == 0 {
		return fmt.Errorf("%w:animation_tracks", ErrRequalificationUnsupported)
	}
	for _, rawTrack := range tracks {
		track, ok := rawTrack.(map[string]any)
		if !ok {
			return ErrAuthoringPayloadInvalid
		}
		keys, ok := track["keys"].([]any)
		if !ok || len(keys) == 0 {
			return fmt.Errorf("%w:animation_track_keys", ErrRequalificationUnsupported)
		}
		for _, rawKey := range keys {
			key, ok := rawKey.(map[string]any)
			if !ok {
				return ErrAuthoringPayloadInvalid
			}
			curve := strings.ToLower(strings.TrimSpace(fmt.Sprint(key["curve"])))
			if curve == "" {
				curve = "linear"
			}
			if curve != "linear" {
				return fmt.Errorf("%w:curve:%s", ErrRequalificationUnsupported, curve)
			}
		}
	}
	return nil
}

func findRequalificationByRevision(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectUUID, revisionUUID uuid.UUID,
) (requalificationRecord, bool, error) {
	var publicID string
	err := pool.QueryRow(ctx, `
		SELECT public_id FROM studio_requalification_requests
		WHERE project_id=$1 AND authoring_revision_id=$2
	`, projectUUID, revisionUUID).Scan(&publicID)
	if errors.Is(err, pgx.ErrNoRows) {
		return requalificationRecord{}, false, nil
	}
	if err != nil {
		return requalificationRecord{}, false, err
	}
	record, err := loadRequalificationByUUID(ctx, pool, projectUUID, publicID)
	return record, err == nil, err
}

func loadRequalification(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID, publicID string,
) (requalificationRecord, error) {
	var projectUUID uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT id FROM studio_projects WHERE public_id=$1`, projectID).Scan(&projectUUID); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return requalificationRecord{}, ErrRequalificationNotFound
		}
		return requalificationRecord{}, err
	}
	return loadRequalificationByUUID(ctx, pool, projectUUID, publicID)
}

func loadRequalificationByUUID(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectUUID uuid.UUID,
	publicID string,
) (requalificationRecord, error) {
	var out requalificationRecord
	var invalidatedRaw []byte
	err := pool.QueryRow(ctx, `
		SELECT rq.id,rq.public_id,rq.project_id,rq.authoring_revision_id,ar.revision_number,
		       p.source_compile_session_id,rq.base_product_revision_id,rq.authoring_artifact_id,
		       rq.attempt_id,rq.state,rq.invalidated_domains,rq.result_product_revision_id,
		       rq.result_compile_session_id,rq.failure_code,rq.created_at,rq.updated_at
		FROM studio_requalification_requests rq
		JOIN studio_projects p ON p.id=rq.project_id
		JOIN studio_authoring_revisions ar ON ar.id=rq.authoring_revision_id
		WHERE rq.project_id=$1 AND rq.public_id=$2
	`, projectUUID, publicID).Scan(
		&out.ID, &out.PublicID, &out.ProjectUUID, &out.RevisionUUID, &out.RevisionNumber,
		&out.BaseSessionID, &out.BaseRevisionID, &out.AuthoringArtifactID,
		&out.AttemptID, &out.State, &invalidatedRaw, &out.ResultRevisionID,
		&out.ResultSessionID, &out.FailureCode, &out.CreatedAt, &out.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return requalificationRecord{}, ErrRequalificationNotFound
	}
	if err != nil {
		return requalificationRecord{}, err
	}
	if err := json.Unmarshal(invalidatedRaw, &out.InvalidatedDomains); err != nil {
		return requalificationRecord{}, err
	}
	return out, nil
}

func syncRequalification(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID string,
	record requalificationRecord,
) (RequalificationStatus, error) {
	if record.State == "PASS" {
		return publicRequalificationStatus(ctx, pool, projectID, record)
	}
	if record.State == "FAILED" {
		return publicRequalificationStatus(ctx, pool, projectID, record)
	}

	var finalState string
	if err := pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, record.AttemptID).Scan(&finalState); err != nil {
		return RequalificationStatus{}, err
	}
	if finalState == "QUALIFIED" {
		var revisionID uuid.UUID
		if err := pool.QueryRow(ctx, `
			SELECT id FROM product_revisions WHERE created_from_attempt_id=$1
		`, record.AttemptID).Scan(&revisionID); err != nil {
			return RequalificationStatus{}, err
		}
		resultSession, err := publicproduct.CreateRequalificationResultSession(
			ctx, pool, record.BaseSessionID, record.AttemptID, revisionID, "studio-requalification",
		)
		if err != nil {
			return RequalificationStatus{}, err
		}
		if _, err := pool.Exec(ctx, `
			UPDATE studio_requalification_requests
			SET state='PASS',result_product_revision_id=$2,result_compile_session_id=$3,failure_code=NULL
			WHERE id=$1 AND state IN ('QUEUED','RUNNING')
		`, record.ID, revisionID, resultSession.SessionID); err != nil {
			return RequalificationStatus{}, err
		}
		return GetRequalification(ctx, pool, projectID, record.PublicID)
	}
	if finalState == "FAILED" {
		code := "PRODUCT_REQUALIFICATION_FAILED"
		if _, err := pool.Exec(ctx, `
			UPDATE studio_requalification_requests SET state='FAILED',failure_code=$2
			WHERE id=$1 AND state IN ('QUEUED','RUNNING')
		`, record.ID, code); err != nil {
			return RequalificationStatus{}, err
		}
		return GetRequalification(ctx, pool, projectID, record.PublicID)
	}

	var executionStatus, stage string
	var errorCode *string
	err := pool.QueryRow(ctx, `
		SELECT status,stage_contract,error_code
		FROM executions WHERE attempt_id=$1
		ORDER BY started_at DESC NULLS LAST,finished_at DESC NULLS LAST
		LIMIT 1
	`, record.AttemptID).Scan(&executionStatus, &stage, &errorCode)
	if errors.Is(err, pgx.ErrNoRows) {
		return publicRequalificationStatus(ctx, pool, projectID, record)
	}
	if err != nil {
		return RequalificationStatus{}, err
	}
	if executionStatus == "FAIL" || executionStatus == "FAILED" || executionStatus == "ERROR" {
		code := "PRODUCT_REQUALIFICATION_CAPABILITY_FAILED"
		if errorCode != nil && strings.TrimSpace(*errorCode) != "" {
			code = *errorCode
		}
		if _, err := pool.Exec(ctx, `
			UPDATE studio_requalification_requests SET state='FAILED',failure_code=$2
			WHERE id=$1 AND state IN ('QUEUED','RUNNING')
		`, record.ID, code); err != nil {
			return RequalificationStatus{}, err
		}
		return GetRequalification(ctx, pool, projectID, record.PublicID)
	}
	if record.State == "QUEUED" {
		if _, err := pool.Exec(ctx, `UPDATE studio_requalification_requests SET state='RUNNING' WHERE id=$1 AND state='QUEUED'`, record.ID); err != nil {
			return RequalificationStatus{}, err
		}
		record.State = "RUNNING"
	}
	status, err := publicRequalificationStatus(ctx, pool, projectID, record)
	if err != nil {
		return RequalificationStatus{}, err
	}
	status.Stage = strings.ToUpper(strings.TrimPrefix(stage, "product."))
	status.Detail = "Requalifying edited product authority through " + status.Stage + "."
	return status, nil
}

func publicRequalificationStatus(
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID string,
	record requalificationRecord,
) (RequalificationStatus, error) {
	out := RequalificationStatus{
		RequalificationID:  record.PublicID,
		ProjectID:          projectID,
		RevisionNumber:     record.RevisionNumber,
		State:              record.State,
		Stage:              "MOTION",
		InvalidatedDomains: append([]string(nil), record.InvalidatedDomains...),
		CreatedAt:          record.CreatedAt,
		UpdatedAt:          record.UpdatedAt,
	}
	switch record.State {
	case "QUEUED":
		out.Detail = "Saved Studio animation revision admitted for Motion and Proof requalification."
	case "RUNNING":
		out.Detail = "Requalifying saved Studio animation revision."
	case "FAILED":
		out.Terminal = true
		out.Stage = "PROOF"
		if record.FailureCode != nil {
			out.Detail = "Requalification failed: " + *record.FailureCode
		} else {
			out.Detail = "Requalification failed."
		}
	case "PASS":
		if record.ResultSessionID == nil || record.ResultRevisionID == nil {
			return RequalificationStatus{}, errors.New("STUDIO_REQUALIFICATION_RESULT_IDENTITY_INCOMPLETE")
		}
		if err := pool.QueryRow(ctx, `
			SELECT public_id FROM public_product_compile_sessions WHERE id=$1
		`, *record.ResultSessionID).Scan(&out.ResultCompileID); err != nil {
			return RequalificationStatus{}, err
		}
		out.Terminal = true
		out.Stage = "FINALIZE"
		out.Detail = "Proof passed. A new qualified ProductRevision was sealed and its clean workspace is being prepared."
	default:
		return RequalificationStatus{}, errors.New("STUDIO_REQUALIFICATION_STATE_INVALID")
	}
	return out, nil
}
