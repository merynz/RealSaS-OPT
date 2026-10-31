package publicproduct

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
)

const MaxWorkspaceBundleBytes = 256 << 20
const workspaceSourceObservationRole = "support:source_observation"

var ErrNoWorkspaceJob = errors.New("PUBLIC_PRODUCT_WORKSPACE_JOB_NOT_AVAILABLE")

type WorkspaceArtifact struct {
	Role          string    `json:"role"`
	ArtifactID    uuid.UUID `json:"artifact_id"`
	ArtifactType  string    `json:"artifact_type"`
	SchemaVersion string    `json:"schema_version"`
	Scope         string    `json:"scope"`
}

type WorkspaceLease struct {
	CompileID         string              `json:"compile_id"`
	ProductRevisionID uuid.UUID           `json:"product_revision_id"`
	LeaseToken        uuid.UUID           `json:"lease_token"`
	LeaseOwner        string              `json:"lease_owner"`
	LeaseExpiry       time.Time           `json:"lease_expires_at"`
	Artifacts         []WorkspaceArtifact `json:"artifacts"`
	Attempt           int                 `json:"attempt"`
}

func ensureWorkspaceMaterialization(
	ctx context.Context,
	pool *pgxpool.Pool,
	sessionID uuid.UUID,
	revisionID uuid.UUID,
) error {
	_, err := pool.Exec(ctx, `
		INSERT INTO public_product_workspace_materializations(
			session_id,product_revision_id,state
		) VALUES ($1,$2,'QUEUED')
		ON CONFLICT (session_id) DO NOTHING
	`, sessionID, revisionID)
	if err != nil {
		return err
	}
	var boundRevision uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT product_revision_id
		FROM public_product_workspace_materializations
		WHERE session_id=$1
	`, sessionID).Scan(&boundRevision); err != nil {
		return err
	}
	if boundRevision != revisionID {
		return errors.New("PUBLIC_PRODUCT_WORKSPACE_REVISION_DRIFT")
	}
	return nil
}

func ClaimWorkspaceJob(
	ctx context.Context,
	pool *pgxpool.Pool,
	owner string,
	leaseDuration time.Duration,
) (WorkspaceLease, error) {
	owner = strings.TrimSpace(owner)
	if pool == nil || owner == "" {
		return WorkspaceLease{}, errors.New("pool and lease owner are required")
	}
	if leaseDuration < time.Minute || leaseDuration > 2*time.Hour {
		return WorkspaceLease{}, errors.New("lease duration must be between one minute and two hours")
	}

	var lease WorkspaceLease
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var sessionID, revisionID uuid.UUID
		var publicID string
		var attemptCount int
		err := tx.QueryRow(ctx, `
			SELECT j.session_id,j.product_revision_id,s.public_id,j.attempt_count
			FROM public_product_workspace_materializations j
			JOIN public_product_compile_sessions s ON s.id=j.session_id
			WHERE (
			    j.state='QUEUED'
			    OR (j.state='LEASED' AND j.lease_expires_at <= now())
			)
			ORDER BY j.created_at,j.session_id
			FOR UPDATE OF j SKIP LOCKED
			LIMIT 1
		`).Scan(&sessionID, &revisionID, &publicID, &attemptCount)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNoWorkspaceJob
		}
		if err != nil {
			return err
		}

		leaseToken := uuid.New()
		leaseExpiry := time.Now().UTC().Add(leaseDuration)
		if _, err := tx.Exec(ctx, `
			UPDATE public_product_workspace_materializations
			SET state='LEASED',lease_token=$2,lease_owner=$3,lease_expires_at=$4,
			    attempt_count=attempt_count+1,last_error_code=NULL
			WHERE session_id=$1
		`, sessionID, leaseToken, owner, leaseExpiry); err != nil {
			return err
		}

		rows, err := tx.Query(ctx, `
			SELECT pra.role,pra.artifact_id,at.name,at.schema_version
			FROM product_revision_artifacts pra
			JOIN artifacts a ON a.id=pra.artifact_id
			JOIN artifact_types at ON at.id=a.artifact_type_id
			WHERE pra.product_revision_id=$1
			ORDER BY pra.role
		`, revisionID)
		if err != nil {
			return err
		}
		artifacts := make([]WorkspaceArtifact, 0, 8)
		for rows.Next() {
			var item WorkspaceArtifact
			if err := rows.Scan(&item.Role, &item.ArtifactID, &item.ArtifactType, &item.SchemaVersion); err != nil {
				rows.Close()
				return err
			}
			item.Scope = "AUTHORITY"
			artifacts = append(artifacts, item)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return err
		}
		rows.Close()
		if len(artifacts) != 7 {
			return fmt.Errorf("PUBLIC_PRODUCT_WORKSPACE_AUTHORITY_ARTIFACT_COUNT_%d", len(artifacts))
		}

		var support WorkspaceArtifact
		support.Role = workspaceSourceObservationRole
		support.Scope = "SUPPORT"
		if err := tx.QueryRow(ctx, `
			SELECT s.source_observation_artifact_id,at.name,at.schema_version
			FROM public_product_compile_sessions s
			JOIN artifacts a ON a.id=s.source_observation_artifact_id
			JOIN artifact_types at ON at.id=a.artifact_type_id
			WHERE s.id=$1 AND s.source_observation_artifact_id IS NOT NULL
		`, sessionID).Scan(&support.ArtifactID, &support.ArtifactType, &support.SchemaVersion); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return errors.New("PUBLIC_PRODUCT_WORKSPACE_SOURCE_OBSERVATION_SUPPORT_REQUIRED")
			}
			return err
		}
		if support.ArtifactType != "RealSaS.SourceObservation" || support.SchemaVersion != "v1" {
			return errors.New("PUBLIC_PRODUCT_WORKSPACE_SOURCE_OBSERVATION_SUPPORT_DRIFT")
		}
		artifacts = append(artifacts, support)

		lease = WorkspaceLease{
			CompileID:         publicID,
			ProductRevisionID: revisionID,
			LeaseToken:        leaseToken,
			LeaseOwner:        owner,
			LeaseExpiry:       leaseExpiry,
			Artifacts:         artifacts,
			Attempt:           attemptCount + 1,
		}
		return nil
	})
	return lease, err
}

func ReadWorkspaceArtifactBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	compileID string,
	leaseToken uuid.UUID,
	role string,
) ([]byte, WorkspaceArtifact, error) {
	if store == nil {
		return nil, WorkspaceArtifact{}, errors.New("artifact store is required")
	}
	session, err := requireLiveWorkspaceLease(ctx, pool, compileID, leaseToken)
	if err != nil {
		return nil, WorkspaceArtifact{}, err
	}
	role = strings.TrimSpace(role)
	if role == "" || len(role) > 150 {
		return nil, WorkspaceArtifact{}, errors.New("PUBLIC_PRODUCT_WORKSPACE_ROLE_INVALID")
	}
	var item WorkspaceArtifact
	if role == workspaceSourceObservationRole {
		item.Role = workspaceSourceObservationRole
		item.Scope = "SUPPORT"
		if err := pool.QueryRow(ctx, `
			SELECT s.source_observation_artifact_id,at.name,at.schema_version
			FROM public_product_compile_sessions s
			JOIN artifacts a ON a.id=s.source_observation_artifact_id
			JOIN artifact_types at ON at.id=a.artifact_type_id
			WHERE s.id=$1 AND s.source_observation_artifact_id IS NOT NULL
		`, session.ID).Scan(&item.ArtifactID, &item.ArtifactType, &item.SchemaVersion); err != nil {
			return nil, WorkspaceArtifact{}, err
		}
		if item.ArtifactType != "RealSaS.SourceObservation" || item.SchemaVersion != "v1" {
			return nil, WorkspaceArtifact{}, errors.New("PUBLIC_PRODUCT_WORKSPACE_SOURCE_OBSERVATION_SUPPORT_DRIFT")
		}
	} else {
		if err := pool.QueryRow(ctx, `
			SELECT pra.role,pra.artifact_id,at.name,at.schema_version
			FROM product_revision_artifacts pra
			JOIN artifacts a ON a.id=pra.artifact_id
			JOIN artifact_types at ON at.id=a.artifact_type_id
			WHERE pra.product_revision_id=$1 AND pra.role=$2
		`, session.ProductRevisionID, role).Scan(
			&item.Role, &item.ArtifactID, &item.ArtifactType, &item.SchemaVersion,
		); err != nil {
			return nil, WorkspaceArtifact{}, err
		}
		item.Scope = "AUTHORITY"
	}
	registered, err := registry.Read(ctx, pool, item.ArtifactID)
	if err != nil {
		return nil, WorkspaceArtifact{}, err
	}
	payload, err := store.GetBytes(ctx, registered.StorageKey)
	if err != nil {
		return nil, WorkspaceArtifact{}, err
	}
	if int64(len(payload)) != registered.SizeBytes {
		return nil, WorkspaceArtifact{}, errors.New("PUBLIC_PRODUCT_WORKSPACE_SOURCE_BYTE_LENGTH_DRIFT")
	}
	return payload, item, nil
}

func CompleteWorkspaceBundleBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	compileID string,
	leaseToken uuid.UUID,
	payload []byte,
	actor string,
) error {
	if store == nil || strings.TrimSpace(actor) == "" {
		return errors.New("artifact store and actor are required")
	}
	if len(payload) == 0 || len(payload) > MaxWorkspaceBundleBytes {
		return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_SIZE_INVALID")
	}
	session, err := requireLiveWorkspaceLease(ctx, pool, compileID, leaseToken)
	if err != nil {
		return err
	}
	if err := validateWorkspaceBundle(payload); err != nil {
		return err
	}
	object, err := store.PutBytes(ctx, payload)
	if err != nil {
		return err
	}
	registered, err := registry.Import(ctx, pool, store, registry.ImportRequest{
		ArtifactType:  "RealSaS.ProductWorkspaceBundleZip",
		SchemaVersion: "v1",
		Object:        object,
		SourceURI:     "product-revision://" + session.ProductRevisionID.String() + "/workspace",
		CreatedBy:     actor,
	})
	if err != nil {
		return err
	}
	result, err := pool.Exec(ctx, `
		UPDATE public_product_workspace_materializations
		SET state='PASS',bundle_artifact_id=$3,lease_token=NULL,lease_owner=NULL,
		    lease_expires_at=NULL,last_error_code=NULL
		WHERE session_id=$1 AND lease_token=$2 AND state='LEASED'
	`, session.ID, leaseToken, registered.ArtifactID)
	if err != nil {
		return err
	}
	if result.RowsAffected() != 1 {
		return errors.New("PUBLIC_PRODUCT_WORKSPACE_LEASE_LOST")
	}
	_, err = pool.Exec(ctx, `
		UPDATE public_product_compile_sessions
		SET state='PASS',current_capability='workspace.materialization',error_code=NULL
		WHERE id=$1
	`, session.ID)
	return err
}

func FailWorkspaceJob(
	ctx context.Context,
	pool *pgxpool.Pool,
	compileID string,
	leaseToken uuid.UUID,
	errorCode string,
	retry bool,
) error {
	errorCode = strings.TrimSpace(errorCode)
	if pool == nil || strings.TrimSpace(compileID) == "" || leaseToken == uuid.Nil || errorCode == "" {
		return errors.New("pool, compile_id, lease_token and error_code are required")
	}
	if len(errorCode) > 200 {
		return errors.New("error_code exceeds 200 characters")
	}
	return persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var sessionID uuid.UUID
		var currentToken uuid.UUID
		var expiresAt time.Time
		if err := tx.QueryRow(ctx, `
			SELECT j.session_id,j.lease_token,j.lease_expires_at
			FROM public_product_workspace_materializations j
			JOIN public_product_compile_sessions s ON s.id=j.session_id
			WHERE s.public_id=$1 AND j.state='LEASED'
			FOR UPDATE OF j
		`, compileID).Scan(&sessionID, &currentToken, &expiresAt); err != nil {
			return err
		}
		if currentToken != leaseToken || time.Now().UTC().After(expiresAt) {
			return errors.New("PUBLIC_PRODUCT_WORKSPACE_LEASE_INVALID_OR_EXPIRED")
		}
		if retry {
			_, err := tx.Exec(ctx, `
				UPDATE public_product_workspace_materializations
				SET state='QUEUED',lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error_code=$2
				WHERE session_id=$1
			`, sessionID, errorCode)
			return err
		}
		if _, err := tx.Exec(ctx, `
			UPDATE public_product_workspace_materializations
			SET state='FAILED',lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error_code=$2
			WHERE session_id=$1
		`, sessionID, errorCode); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
			UPDATE public_product_compile_sessions
			SET state='FAILED',error_code=$2,current_capability='workspace.materialization'
			WHERE id=$1
		`, sessionID, errorCode)
		return err
	})
}

func WorkspaceBundleBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	compileID string,
) ([]byte, error) {
	if store == nil {
		return nil, errors.New("artifact store is required")
	}
	session, err := loadSession(ctx, pool, compileID)
	if err != nil {
		return nil, err
	}
	var artifactID uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT bundle_artifact_id
		FROM public_product_workspace_materializations
		WHERE session_id=$1 AND state='PASS'
	`, session.ID).Scan(&artifactID); err != nil {
		return nil, err
	}
	registered, err := registry.Read(ctx, pool, artifactID)
	if err != nil {
		return nil, err
	}
	payload, err := store.GetBytes(ctx, registered.StorageKey)
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) != registered.SizeBytes {
		return nil, errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_BYTE_LENGTH_DRIFT")
	}
	return payload, nil
}

func requireLiveWorkspaceLease(
	ctx context.Context,
	pool *pgxpool.Pool,
	compileID string,
	leaseToken uuid.UUID,
) (sessionRecord, error) {
	if pool == nil || strings.TrimSpace(compileID) == "" || leaseToken == uuid.Nil {
		return sessionRecord{}, errors.New("pool, compile_id and lease_token are required")
	}
	session, err := loadSession(ctx, pool, compileID)
	if err != nil {
		return sessionRecord{}, err
	}
	var currentToken uuid.UUID
	var expiresAt time.Time
	if err := pool.QueryRow(ctx, `
		SELECT lease_token,lease_expires_at
		FROM public_product_workspace_materializations
		WHERE session_id=$1 AND state='LEASED'
	`, session.ID).Scan(&currentToken, &expiresAt); err != nil {
		return sessionRecord{}, err
	}
	if currentToken != leaseToken || time.Now().UTC().After(expiresAt) {
		return sessionRecord{}, errors.New("PUBLIC_PRODUCT_WORKSPACE_LEASE_INVALID_OR_EXPIRED")
	}
	return session, nil
}

func validateWorkspaceBundle(payload []byte) error {
	reader, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_ZIP_INVALID")
	}
	required := map[string]bool{
		"bundle_manifest.json":                  false,
		"puppet/canonical_puppet_graph_v3.json": false,
		"proof/product_proof_bundle_ir.json":    false,
	}
	for _, file := range reader.File {
		name := strings.ReplaceAll(file.Name, "\\", "/")
		if name == "" || strings.HasPrefix(name, "/") || strings.Contains(name, "../") {
			return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_PATH_INVALID")
		}
		if _, ok := required[name]; ok {
			required[name] = true
		}
		if file.UncompressedSize64 > MaxWorkspaceBundleBytes {
			return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_ENTRY_TOO_LARGE")
		}
		if file.FileInfo().Mode().IsRegular() {
			r, err := file.Open()
			if err != nil {
				return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_ENTRY_INVALID")
			}
			_, copyErr := io.Copy(io.Discard, io.LimitReader(r, MaxWorkspaceBundleBytes+1))
			closeErr := r.Close()
			if copyErr != nil || closeErr != nil {
				return errors.New("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_ENTRY_INVALID")
			}
		}
	}
	for name, present := range required {
		if !present {
			return fmt.Errorf("PUBLIC_PRODUCT_WORKSPACE_BUNDLE_REQUIRED_FILE_MISSING:%s", name)
		}
	}
	return nil
}