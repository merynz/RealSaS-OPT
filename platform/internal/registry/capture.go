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

// CaptureRequest registers bytes produced by an internal product surface but
// not yet qualified as canonical product authority. Unlike Import, Capture does
// not call the payload EXTERNAL_INPUT; callers must name the precise internal
// authority class and producer contract. Capture never creates a qualification.
type CaptureRequest struct {
	ArtifactType       string
	SchemaVersion      string
	Payload            []byte
	ProducerContract   string
	AuthorityClass     string
	SemanticParameters map[string]any
	CreatedBy          string
}

type Captured struct {
	ArtifactID     uuid.UUID `json:"artifact_id"`
	SemanticSHA256 string    `json:"semantic_sha256"`
	ContentSHA256  string    `json:"content_sha256"`
	Reused         bool      `json:"reused"`
}

func CaptureBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	req CaptureRequest,
) (Captured, error) {
	if pool == nil || store == nil {
		return Captured{}, errors.New("postgres pool and artifact store are required")
	}
	req.ArtifactType = strings.TrimSpace(req.ArtifactType)
	req.SchemaVersion = strings.TrimSpace(req.SchemaVersion)
	req.ProducerContract = strings.TrimSpace(req.ProducerContract)
	req.AuthorityClass = strings.TrimSpace(req.AuthorityClass)
	req.CreatedBy = strings.TrimSpace(req.CreatedBy)
	if req.ArtifactType == "" || req.SchemaVersion == "" || req.ProducerContract == "" ||
		req.AuthorityClass == "" || req.CreatedBy == "" || len(req.Payload) == 0 {
		return Captured{}, errors.New("artifact type, schema, payload, producer contract, authority class and created_by are required")
	}
	params := make(map[string]any, len(req.SemanticParameters)+2)
	for key, value := range req.SemanticParameters {
		params[key] = value
	}
	params["authority_class"] = req.AuthorityClass
	params["qualification_granted"] = false

	object, err := store.PutBytes(ctx, req.Payload)
	if err != nil {
		return Captured{}, err
	}
	descriptor := domain.ArtifactSemanticDescriptor{
		ArtifactType:         req.ArtifactType,
		SchemaVersion:        req.SchemaVersion,
		ProducerContract:     req.ProducerContract,
		ImplementationSHA256: artifactstore.HashBytes([]byte(req.ProducerContract)),
		PolicySHA256:         artifactstore.HashBytes([]byte("INTERNAL_CAPTURE__NO_QUALIFICATION__" + req.AuthorityClass)),
		Inputs:               []domain.ArtifactInputIdentity{},
		SemanticParameters:   params,
	}
	semanticSHA, err := descriptor.SemanticSHA256()
	if err != nil {
		return Captured{}, err
	}
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return Captured{}, err
	}

	var out Captured
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		typeID, err := ensureArtifactType(ctx, tx, req.ArtifactType, req.SchemaVersion)
		if err != nil {
			return err
		}
		artifactID := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:capture:"+semanticSHA))
		tag, err := tx.Exec(ctx, `
			INSERT INTO artifacts(
				id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now())
			ON CONFLICT (artifact_type_id,semantic_sha256) DO NOTHING
		`, artifactID, typeID, semanticSHA, object.ContentSHA256, object.StorageKey, object.SizeBytes,
			descriptor.ProducerContract, descriptor.ImplementationSHA256, descriptor.PolicySHA256, paramsJSON)
		if err != nil {
			return err
		}

		var actualID uuid.UUID
		var contentSHA, storageKey string
		var size int64
		if err := tx.QueryRow(ctx, `
			SELECT id,content_sha256,storage_key,size_bytes
			FROM artifacts
			WHERE artifact_type_id=$1 AND semantic_sha256=$2
			FOR SHARE
		`, typeID, semanticSHA).Scan(&actualID, &contentSHA, &storageKey, &size); err != nil {
			return err
		}
		if contentSHA != object.ContentSHA256 || storageKey != object.StorageKey || size != object.SizeBytes {
			return errors.New("CAPTURED_ARTIFACT_IDENTITY_DRIFT")
		}
		audit, _ := json.Marshal(map[string]any{
			"artifact_id":       actualID,
			"artifact_type":     req.ArtifactType,
			"schema_version":    req.SchemaVersion,
			"semantic_sha256":   semanticSHA,
			"content_sha256":    object.ContentSHA256,
			"authority_class":   req.AuthorityClass,
			"producer_contract": req.ProducerContract,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,artifact_id,payload)
			VALUES ($1,'INTERNAL_ARTIFACT_CAPTURED',$2,$3)
		`, req.CreatedBy, actualID, audit); err != nil {
			return err
		}
		out = Captured{
			ArtifactID:     actualID,
			SemanticSHA256: semanticSHA,
			ContentSHA256:  object.ContentSHA256,
			Reused:         tag.RowsAffected() == 0,
		}
		return nil
	})
	return out, err
}
