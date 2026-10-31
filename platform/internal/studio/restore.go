package studio

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrRevisionNotFound = errors.New("STUDIO_REVISION_NOT_FOUND")

type RestoreRequest struct {
	ExpectedLockVersion int64  `json:"expected_lock_version"`
	RevisionNumber      int64  `json:"revision_number"`
	CreatedBy           string `json:"created_by"`
}

// RestoreRevision never rewinds studio_project_heads to an old row. It copies
// the selected historical authoring payload into a new immutable revision and
// advances the head under the same optimistic lock used by normal saves. This
// preserves the full audit chain and keeps requalification fail-closed.
func RestoreRevision(ctx context.Context, pool *pgxpool.Pool, projectID string, req RestoreRequest) (Revision, error) {
	if pool == nil || !projectIDPattern.MatchString(projectID) || req.ExpectedLockVersion < 0 || req.RevisionNumber <= 0 {
		return Revision{}, ErrRevisionNotFound
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

	var targetPayload []byte
	var targetSHA string
	var targetProductRevision uuid.UUID
	var targetManifest string
	err = tx.QueryRow(ctx, `
		SELECT authoring_payload, payload_sha256, source_product_revision_id, source_manifest_sha256
		FROM studio_authoring_revisions
		WHERE project_id=$1 AND revision_number=$2
	`, identity.ID, req.RevisionNumber).Scan(&targetPayload, &targetSHA, &targetProductRevision, &targetManifest)
	if errors.Is(err, pgx.ErrNoRows) {
		return Revision{}, ErrRevisionNotFound
	}
	if err != nil {
		return Revision{}, err
	}
	if targetProductRevision != identity.ProductRevisionID || targetManifest != identity.ManifestSHA256 {
		return Revision{}, errors.New("STUDIO_REVISION_BASE_DRIFT")
	}
	canonical, canonicalSHA, err := canonicalAuthoringPayload(json.RawMessage(targetPayload))
	if err != nil {
		return Revision{}, err
	}
	if canonicalSHA != targetSHA {
		return Revision{}, errors.New("STUDIO_REVISION_PAYLOAD_DRIFT")
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
		if currentSHA == canonicalSHA {
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
		identity.ManifestSHA256, canonicalSHA, string(canonical), createdBy).Scan(&createdAt); err != nil {
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
		LockVersion: identity.LockVersion + 1, PayloadSHA256: canonicalSHA,
		RequiresRequalification: true, CreatedAt: createdAt,
	}, nil
}
