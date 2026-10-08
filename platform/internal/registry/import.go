package registry

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

// ImportRequest registers exact external bytes, never a stage cache hit or a
// qualification. The Engine must execute the consuming contract to mint those.
type ImportRequest struct {
	ArtifactType  string               `json:"artifact_type"`
	SchemaVersion string               `json:"schema_version"`
	Object        artifactstore.Object `json:"object"`
	SourceURI     string               `json:"source_uri"`
	CreatedBy     string               `json:"created_by"`
}

type Imported struct {
	ArtifactID     uuid.UUID `json:"artifact_id"`
	SemanticSHA256 string    `json:"semantic_sha256"`
	Reused         bool      `json:"reused"`
}

func Import(ctx context.Context, pool *pgxpool.Pool, store artifactstore.Store, req ImportRequest) (Imported, error) {
	if strings.TrimSpace(req.ArtifactType) == "" || strings.TrimSpace(req.SchemaVersion) == "" || strings.TrimSpace(req.SourceURI) == "" || strings.TrimSpace(req.CreatedBy) == "" {
		return Imported{}, errors.New("artifact type, schema, source_uri and created_by are required")
	}
	if err := artifactstore.ValidateSHA256(req.Object.ContentSHA256); err != nil {
		return Imported{}, err
	}
	key, _ := artifactstore.CASKey(req.Object.ContentSHA256, "")
	if req.Object.StorageKey != key || req.Object.SizeBytes < 0 {
		return Imported{}, errors.New("import requires canonical CAS identity and nonnegative size")
	}
	if store == nil {
		return Imported{}, errors.New("artifact store is required")
	}
	if err := store.Verify(ctx, req.Object); err != nil {
		return Imported{}, err
	}
	descriptor := domain.ArtifactSemanticDescriptor{
		ArtifactType: req.ArtifactType, SchemaVersion: req.SchemaVersion,
		ProducerContract:     "RealSaS.ExternalEvidenceImport.v1",
		ImplementationSHA256: artifactstore.HashBytes([]byte("RealSaS.ExternalEvidenceImport.v1")),
		PolicySHA256:         artifactstore.HashBytes([]byte("RAW_INPUT_ONLY__NO_QUALIFICATION__NO_STAGE_REUSE")),
		Inputs:               []domain.ArtifactInputIdentity{},
		SemanticParameters:   map[string]any{"content_sha256": req.Object.ContentSHA256, "size_bytes": req.Object.SizeBytes, "authority_class": "EXTERNAL_INPUT"},
	}
	sha, err := descriptor.SemanticSHA256()
	if err != nil {
		return Imported{}, err
	}
	params, err := json.Marshal(descriptor.SemanticParameters)
	if err != nil {
		return Imported{}, err
	}
	var out Imported
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		typeID, err := ensureArtifactType(ctx, tx, req.ArtifactType, req.SchemaVersion)
		if err != nil {
			return err
		}
		id := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:import:"+sha))
		tag, err := tx.Exec(ctx, `INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now()) ON CONFLICT (artifact_type_id,semantic_sha256) DO NOTHING`,
			id, typeID, sha, req.Object.ContentSHA256, req.Object.StorageKey, req.Object.SizeBytes, descriptor.ProducerContract, descriptor.ImplementationSHA256, descriptor.PolicySHA256, params)
		if err != nil {
			return err
		}
		var actualID uuid.UUID
		var content, key string
		var size int64
		if err := tx.QueryRow(ctx, `SELECT id,content_sha256,storage_key,size_bytes FROM artifacts WHERE artifact_type_id=$1 AND semantic_sha256=$2 FOR SHARE`, typeID, sha).Scan(&actualID, &content, &key, &size); err != nil {
			return err
		}
		if content != req.Object.ContentSHA256 || key != req.Object.StorageKey || size != req.Object.SizeBytes {
			return errors.New("IMPORTED_ARTIFACT_IDENTITY_DRIFT")
		}
		audit, _ := json.Marshal(map[string]any{"artifact_id": actualID, "source_uri": req.SourceURI, "object": req.Object, "semantic_sha256": sha, "authority_class": "EXTERNAL_INPUT"})
		if _, err := tx.Exec(ctx, `INSERT INTO audit_events(actor,action,payload) VALUES ($1,'EXTERNAL_EVIDENCE_IMPORTED',$2)`, req.CreatedBy, audit); err != nil {
			return err
		}
		out = Imported{ArtifactID: actualID, SemanticSHA256: sha, Reused: tag.RowsAffected() == 0}
		return nil
	})
	return out, err
}
