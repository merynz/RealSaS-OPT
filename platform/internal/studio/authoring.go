package studio

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	AuthoringLayerSchema = "RealSaS.UserPuppetEditLayer.v3"
	maxAuthoringBytes    = 4 << 20
	maxRevisionHistory   = 200
)

var (
	ErrProjectNotFound         = errors.New("STUDIO_PROJECT_NOT_FOUND")
	ErrCompileNotReady         = errors.New("STUDIO_COMPILE_NOT_READY")
	ErrRevisionConflict        = errors.New("STUDIO_REVISION_CONFLICT")
	ErrAuthoringPayloadInvalid = errors.New("STUDIO_AUTHORING_PAYLOAD_INVALID")
)

var (
	compileIDPattern = regexp.MustCompile(`^cmp_[0-9a-f]{32}$`)
	projectIDPattern = regexp.MustCompile(`^stp_[0-9a-f]{32}$`)
)

type Project struct {
	ProjectID               string    `json:"project_id"`
	CompileID               string    `json:"compile_id"`
	DisplayName             string    `json:"display_name"`
	State                   string    `json:"state"`
	BaseManifestSHA256      string    `json:"base_manifest_sha256"`
	HeadRevisionNumber      int64     `json:"head_revision_number"`
	LockVersion             int64     `json:"lock_version"`
	RequiresRequalification bool      `json:"requires_requalification"`
	CreatedAt               time.Time `json:"created_at"`
	UpdatedAt               time.Time `json:"updated_at"`
}

type SaveRequest struct {
	ExpectedLockVersion int64           `json:"expected_lock_version"`
	AuthoringPayload    json.RawMessage `json:"authoring_payload"`
	CreatedBy           string          `json:"created_by"`
}

type Revision struct {
	ProjectID               string    `json:"project_id"`
	RevisionNumber          int64     `json:"revision_number"`
	LockVersion             int64     `json:"lock_version"`
	PayloadSHA256           string    `json:"payload_sha256"`
	RequiresRequalification bool      `json:"requires_requalification"`
	CreatedAt               time.Time `json:"created_at"`
	Idempotent              bool      `json:"idempotent"`
}

type Head struct {
	ProjectID               string          `json:"project_id"`
	CompileID               string          `json:"compile_id"`
	RevisionNumber          int64           `json:"revision_number"`
	LockVersion             int64           `json:"lock_version"`
	PayloadSHA256           string          `json:"payload_sha256,omitempty"`
	AuthoringPayload        json.RawMessage `json:"authoring_payload"`
	RequiresRequalification bool            `json:"requires_requalification"`
	UpdatedAt               time.Time       `json:"updated_at"`
}

type RevisionSummary struct {
	RevisionNumber          int64     `json:"revision_number"`
	PayloadSHA256           string    `json:"payload_sha256"`
	RequiresRequalification bool      `json:"requires_requalification"`
	CreatedBy               string    `json:"created_by"`
	CreatedAt               time.Time `json:"created_at"`
	IsHead                  bool      `json:"is_head"`
}

type projectIdentity struct {
	ID                uuid.UUID
	ProductRevisionID uuid.UUID
	ManifestSHA256    string
	HeadRevisionID    *uuid.UUID
	HeadRevision      int64
	LockVersion       int64
}

func EnsureProject(ctx context.Context, pool *pgxpool.Pool, compileID, displayName, createdBy string) (Project, error) {
	compileID = strings.TrimSpace(compileID)
	if pool == nil || !compileIDPattern.MatchString(compileID) {
		return Project{}, ErrCompileNotReady
	}
	displayName = strings.TrimSpace(displayName)
	if displayName == "" {
		displayName = "Untitled Character"
	}
	if len(displayName) > 200 {
		displayName = displayName[:200]
	}
	createdBy = normalizedActor(createdBy)

	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return Project{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var sessionID, productRevisionID uuid.UUID
	var manifestSHA string
	err = tx.QueryRow(ctx, `
		SELECT s.id, s.product_revision_id, pr.manifest_sha256
		FROM public_product_compile_sessions s
		JOIN product_revisions pr ON pr.id = s.product_revision_id
		JOIN public_product_workspace_materializations w
		  ON w.session_id = s.id AND w.product_revision_id = pr.id
		WHERE s.public_id=$1 AND s.state='PASS' AND w.state='PASS'
	`, compileID).Scan(&sessionID, &productRevisionID, &manifestSHA)
	if errors.Is(err, pgx.ErrNoRows) {
		return Project{}, ErrCompileNotReady
	}
	if err != nil {
		return Project{}, err
	}

	projectUUID := uuid.New()
	publicID := newProjectID()
	if _, err := tx.Exec(ctx, `
		INSERT INTO studio_projects(
			id, public_id, source_compile_session_id, product_revision_id,
			display_name, state, created_by
		)
		VALUES ($1,$2,$3,$4,$5,'ACTIVE',$6)
		ON CONFLICT (source_compile_session_id) DO NOTHING
	`, projectUUID, publicID, sessionID, productRevisionID, displayName, createdBy); err != nil {
		return Project{}, err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO studio_project_heads(project_id, authoring_revision_id, lock_version)
		SELECT id, NULL, 0 FROM studio_projects WHERE source_compile_session_id=$1
		ON CONFLICT (project_id) DO NOTHING
	`, sessionID); err != nil {
		return Project{}, err
	}

	project, err := loadProjectBySession(ctx, tx, sessionID)
	if err != nil {
		return Project{}, err
	}
	if project.BaseManifestSHA256 != manifestSHA {
		return Project{}, errors.New("STUDIO_BASE_MANIFEST_DRIFT")
	}
	if err := tx.Commit(ctx); err != nil {
		return Project{}, err
	}
	return project, nil
}

func GetProject(ctx context.Context, pool *pgxpool.Pool, projectID string) (Project, error) {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || !projectIDPattern.MatchString(projectID) {
		return Project{}, ErrProjectNotFound
	}
	return loadProjectByPublicID(ctx, pool, projectID)
}

func GetHead(ctx context.Context, pool *pgxpool.Pool, projectID string) (Head, error) {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || !projectIDPattern.MatchString(projectID) {
		return Head{}, ErrProjectNotFound
	}
	var out Head
	var payload []byte
	err := pool.QueryRow(ctx, `
		SELECT p.public_id, s.public_id, COALESCE(ar.revision_number,0), h.lock_version,
		       COALESCE(ar.payload_sha256,''), COALESCE(ar.authoring_payload,'null'::jsonb),
		       (ar.id IS NOT NULL), h.updated_at
		FROM studio_projects p
		JOIN public_product_compile_sessions s ON s.id=p.source_compile_session_id
		JOIN studio_project_heads h ON h.project_id=p.id
		LEFT JOIN studio_authoring_revisions ar ON ar.id=h.authoring_revision_id
		WHERE p.public_id=$1
	`, projectID).Scan(
		&out.ProjectID, &out.CompileID, &out.RevisionNumber, &out.LockVersion,
		&out.PayloadSHA256, &payload, &out.RequiresRequalification, &out.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Head{}, ErrProjectNotFound
	}
	if err != nil {
		return Head{}, err
	}
	out.AuthoringPayload = json.RawMessage(payload)
	return out, nil
}

func ListRevisions(ctx context.Context, pool *pgxpool.Pool, projectID string, limit int) ([]RevisionSummary, error) {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || !projectIDPattern.MatchString(projectID) {
		return nil, ErrProjectNotFound
	}
	if limit <= 0 {
		limit = 50
	}
	if limit > maxRevisionHistory {
		limit = maxRevisionHistory
	}
	rows, err := pool.Query(ctx, `
		SELECT ar.revision_number, ar.payload_sha256, ar.requires_requalification,
		       ar.created_by, ar.created_at, (h.authoring_revision_id=ar.id)
		FROM studio_projects p
		JOIN studio_project_heads h ON h.project_id=p.id
		JOIN studio_authoring_revisions ar ON ar.project_id=p.id
		WHERE p.public_id=$1
		ORDER BY ar.revision_number DESC
		LIMIT $2
	`, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]RevisionSummary, 0, limit)
	for rows.Next() {
		var item RevisionSummary
		if err := rows.Scan(
			&item.RevisionNumber, &item.PayloadSHA256, &item.RequiresRequalification,
			&item.CreatedBy, &item.CreatedAt, &item.IsHead,
		); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		if _, err := GetProject(ctx, pool, projectID); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func SaveRevision(ctx context.Context, pool *pgxpool.Pool, projectID string, req SaveRequest) (Revision, error) {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || !projectIDPattern.MatchString(projectID) || req.ExpectedLockVersion < 0 {
		return Revision{}, ErrProjectNotFound
	}
	payload, payloadSHA, err := canonicalAuthoringPayload(req.AuthoringPayload)
	if err != nil {
		return Revision{}, err
	}
	createdBy := normalizedActor(req.CreatedBy)

	tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return Revision{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	identity, err := loadProjectIdentityForUpdate(ctx, tx, projectID)
	if err != nil {
		return Revision{}, err
	}
	if identity.LockVersion != req.ExpectedLockVersion {
		return Revision{}, ErrRevisionConflict
	}

	if identity.HeadRevisionID != nil {
		var currentSHA string
		var currentCreated time.Time
		if err := tx.QueryRow(ctx, `
			SELECT payload_sha256, created_at
			FROM studio_authoring_revisions
			WHERE id=$1 AND project_id=$2
		`, *identity.HeadRevisionID, identity.ID).Scan(&currentSHA, &currentCreated); err != nil {
			return Revision{}, err
		}
		if currentSHA == payloadSHA {
			if err := tx.Commit(ctx); err != nil {
				return Revision{}, err
			}
			return Revision{
				ProjectID: projectID, RevisionNumber: identity.HeadRevision,
				LockVersion: identity.LockVersion, PayloadSHA256: currentSHA,
				RequiresRequalification: true, CreatedAt: currentCreated, Idempotent: true,
			}, nil
		}
	}

	revisionID := uuid.New()
	nextRevision := identity.HeadRevision + 1
	var parent any
	if identity.HeadRevisionID != nil {
		parent = *identity.HeadRevisionID
	}
	var createdAt time.Time
	if err := tx.QueryRow(ctx, `
		INSERT INTO studio_authoring_revisions(
			id, project_id, revision_number, parent_revision_id,
			source_product_revision_id, source_manifest_sha256,
			payload_sha256, authoring_payload, requires_requalification, created_by
		)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,true,$9)
		RETURNING created_at
	`, revisionID, identity.ID, nextRevision, parent, identity.ProductRevisionID,
		identity.ManifestSHA256, payloadSHA, string(payload), createdBy).Scan(&createdAt); err != nil {
		return Revision{}, err
	}
	command, err := tx.Exec(ctx, `
		UPDATE studio_project_heads
		SET authoring_revision_id=$2, lock_version=lock_version+1, updated_at=now()
		WHERE project_id=$1 AND lock_version=$3
	`, identity.ID, revisionID, identity.LockVersion)
	if err != nil {
		return Revision{}, err
	}
	if command.RowsAffected() != 1 {
		return Revision{}, ErrRevisionConflict
	}
	if err := tx.Commit(ctx); err != nil {
		return Revision{}, err
	}
	return Revision{
		ProjectID: projectID, RevisionNumber: nextRevision,
		LockVersion: identity.LockVersion + 1, PayloadSHA256: payloadSHA,
		RequiresRequalification: true, CreatedAt: createdAt,
	}, nil
}

func canonicalAuthoringPayload(raw json.RawMessage) ([]byte, string, error) {
	if len(raw) == 0 || len(raw) > maxAuthoringBytes {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil || value == nil {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	if schema, _ := value["schema_version"].(string); schema != AuthoringLayerSchema {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	if authority, _ := value["authority_class"].(string); authority != "OPTIONAL_USER_AUTHORING_LAYER" {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	canPromote, ok := value["can_promote_product_directly"].(bool)
	if !ok || canPromote {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	requiresRequal, ok := value["requires_compiler_requalification"].(bool)
	if !ok || !requiresRequal {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	stateHash, _ := value["source_product_state_hash"].(string)
	if len(stateHash) != 64 {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	metadata, ok := value["metadata"].(map[string]any)
	if !ok {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	mutated, ok := metadata["canonical_bundle_mutated"].(bool)
	if !ok || mutated {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	canonical, err := json.Marshal(value)
	if err != nil || len(canonical) > maxAuthoringBytes {
		return nil, "", ErrAuthoringPayloadInvalid
	}
	digest := sha256.Sum256(canonical)
	return canonical, hex.EncodeToString(digest[:]), nil
}

func loadProjectBySession(ctx context.Context, q interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}, sessionID uuid.UUID) (Project, error) {
	return scanProject(q.QueryRow(ctx, projectSelect+` WHERE p.source_compile_session_id=$1`, sessionID))
}

func loadProjectByPublicID(ctx context.Context, q interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}, publicID string) (Project, error) {
	out, err := scanProject(q.QueryRow(ctx, projectSelect+` WHERE p.public_id=$1`, publicID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Project{}, ErrProjectNotFound
	}
	return out, err
}

const projectSelect = `
	SELECT p.public_id, s.public_id, p.display_name, p.state, pr.manifest_sha256,
	       h.lock_version, COALESCE(ar.revision_number,0),
	       (h.authoring_revision_id IS NOT NULL), p.created_at, p.updated_at
	FROM studio_projects p
	JOIN public_product_compile_sessions s ON s.id=p.source_compile_session_id
	JOIN product_revisions pr ON pr.id=p.product_revision_id
	JOIN studio_project_heads h ON h.project_id=p.id
	LEFT JOIN studio_authoring_revisions ar ON ar.id=h.authoring_revision_id
`

func scanProject(row pgx.Row) (Project, error) {
	var out Project
	if err := row.Scan(
		&out.ProjectID, &out.CompileID, &out.DisplayName, &out.State,
		&out.BaseManifestSHA256, &out.LockVersion, &out.HeadRevisionNumber,
		&out.RequiresRequalification, &out.CreatedAt, &out.UpdatedAt,
	); err != nil {
		return Project{}, err
	}
	return out, nil
}

func loadProjectIdentityForUpdate(ctx context.Context, tx pgx.Tx, publicID string) (projectIdentity, error) {
	var out projectIdentity
	err := tx.QueryRow(ctx, `
		SELECT p.id, p.product_revision_id, pr.manifest_sha256,
		       h.authoring_revision_id, h.lock_version, COALESCE(ar.revision_number,0)
		FROM studio_projects p
		JOIN product_revisions pr ON pr.id=p.product_revision_id
		JOIN studio_project_heads h ON h.project_id=p.id
		LEFT JOIN studio_authoring_revisions ar ON ar.id=h.authoring_revision_id
		WHERE p.public_id=$1 AND p.state='ACTIVE'
		FOR UPDATE OF h
	`, publicID).Scan(
		&out.ID, &out.ProductRevisionID, &out.ManifestSHA256,
		&out.HeadRevisionID, &out.LockVersion, &out.HeadRevision,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return projectIdentity{}, ErrProjectNotFound
	}
	return out, err
}

func normalizedActor(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		value = "product-gateway"
	}
	if len(value) > 300 {
		value = value[:300]
	}
	return value
}

func newProjectID() string {
	return "stp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}
