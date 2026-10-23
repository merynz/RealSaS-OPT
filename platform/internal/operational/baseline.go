package operational

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
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type BaselineKind string

const (
	BaselineProductRevision          BaselineKind = "PRODUCT_REVISION"
	BaselineLegacyEngineeringWitness BaselineKind = "LEGACY_ENGINEERING_WITNESS"
)

var ErrBaselineNotFound = errors.New("subject baseline not found")

type LegacyWitnessRegisterRequest struct {
	SubjectID        uuid.UUID      `json:"subject_id"`
	WitnessArtifactID uuid.UUID      `json:"witness_artifact_id"`
	SourceRef        string         `json:"source_ref"`
	SourceSHA256     string         `json:"source_sha256"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	CreatedBy        string         `json:"created_by"`
}

type LegacyWitnessRecord struct {
	ID                            uuid.UUID      `json:"id"`
	SubjectID                     uuid.UUID      `json:"subject_id"`
	WitnessArtifactID             uuid.UUID      `json:"witness_artifact_id"`
	ArtifactSemanticSHA256        string         `json:"artifact_semantic_sha256"`
	ArtifactContentSHA256         string         `json:"artifact_content_sha256"`
	ArtifactStorageKey            string         `json:"artifact_storage_key"`
	SourceRef                     string         `json:"source_ref"`
	SourceSHA256                  string         `json:"source_sha256"`
	Metadata                      map[string]any `json:"metadata"`
	SupersededByProductRevisionID *uuid.UUID     `json:"superseded_by_product_revision_id,omitempty"`
	CreatedBy                     string         `json:"created_by"`
	CreatedAt                     time.Time      `json:"created_at"`
	Reused                        bool           `json:"reused,omitempty"`
}

type Baseline struct {
	Schema        string               `json:"schema"`
	SubjectID     uuid.UUID            `json:"subject_id"`
	SubjectSlug   string               `json:"subject_slug"`
	Kind          BaselineKind         `json:"kind"`
	Product       *CurrentProduct      `json:"product,omitempty"`
	LegacyWitness *LegacyWitnessRecord `json:"legacy_witness,omitempty"`
}

func RegisterLegacyWitness(ctx context.Context, pool *pgxpool.Pool, raw LegacyWitnessRegisterRequest) (LegacyWitnessRecord, error) {
	req := raw
	req.SourceRef = strings.TrimSpace(req.SourceRef)
	req.SourceSHA256 = strings.TrimSpace(req.SourceSHA256)
	req.CreatedBy = strings.TrimSpace(req.CreatedBy)
	if req.Metadata == nil {
		req.Metadata = map[string]any{}
	}
	if req.SubjectID == uuid.Nil || req.WitnessArtifactID == uuid.Nil || req.SourceRef == "" || req.CreatedBy == "" {
		return LegacyWitnessRecord{}, errors.New("subject_id, witness_artifact_id, source_ref and created_by are required")
	}
	if err := artifactstore.ValidateSHA256(req.SourceSHA256); err != nil {
		return LegacyWitnessRecord{}, err
	}
	metadata, err := json.Marshal(req.Metadata)
	if err != nil {
		return LegacyWitnessRecord{}, err
	}

	var out LegacyWitnessRecord
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var subject uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", req.SubjectID).Scan(&subject); err != nil {
			return fmt.Errorf("legacy witness subject: %w", err)
		}
		var typeName, schema, contentSHA string
		if err := tx.QueryRow(ctx, `
			SELECT t.name,t.schema_version,a.content_sha256
			FROM artifacts a
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE a.id=$1
			FOR SHARE OF a,t
		`, req.WitnessArtifactID).Scan(&typeName, &schema, &contentSHA); err != nil {
			return fmt.Errorf("legacy witness artifact: %w", err)
		}
		if typeName != "RealSaS.ProductNearestWitnessInventory" || schema != "v1" {
			return fmt.Errorf("LEGACY_WITNESS_ARTIFACT_TYPE_MISMATCH:%s:%s", typeName, schema)
		}
		if contentSHA != req.SourceSHA256 {
			return errors.New("LEGACY_WITNESS_CONTENT_SHA_MISMATCH")
		}

		var existingID uuid.UUID
		err := tx.QueryRow(ctx, `
			SELECT id FROM legacy_engineering_witnesses
			WHERE subject_id=$1 AND witness_artifact_id=$2
		`, req.SubjectID, req.WitnessArtifactID).Scan(&existingID)
		if err == nil {
			record, err := loadLegacyWitnessTx(ctx, tx, existingID)
			if err != nil {
				return err
			}
			record.Reused = true
			out = record
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		var activeID uuid.UUID
		err = tx.QueryRow(ctx, `
			SELECT id FROM legacy_engineering_witnesses
			WHERE subject_id=$1 AND superseded_by_product_revision_id IS NULL
			FOR UPDATE
		`, req.SubjectID).Scan(&activeID)
		if err == nil {
			return fmt.Errorf("LEGACY_WITNESS_ACTIVE_ALREADY_EXISTS:%s", activeID)
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		id := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO legacy_engineering_witnesses
			(id,subject_id,witness_artifact_id,source_ref,source_sha256,metadata,created_by)
			VALUES ($1,$2,$3,$4,$5,$6,$7)
		`, id, req.SubjectID, req.WitnessArtifactID, req.SourceRef, req.SourceSHA256, metadata, req.CreatedBy); err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{
			"legacy_witness_id": id.String(),
			"witness_artifact_id": req.WitnessArtifactID.String(),
			"source_ref": req.SourceRef,
			"source_sha256": req.SourceSHA256,
			"authority_class": "ENGINEERING_REGRESSION_WITNESS__NOT_PRODUCT_AUTHORITY",
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,artifact_id,payload)
			VALUES ($1,'LEGACY_ENGINEERING_WITNESS_REGISTERED',$2,$3,$4)
		`, req.CreatedBy, req.SubjectID, req.WitnessArtifactID, audit); err != nil {
			return err
		}
		record, err := loadLegacyWitnessTx(ctx, tx, id)
		if err != nil {
			return err
		}
		out = record
		return nil
	})
	return out, err
}

func loadLegacyWitnessTx(ctx context.Context, tx pgx.Tx, id uuid.UUID) (LegacyWitnessRecord, error) {
	var out LegacyWitnessRecord
	var metadata []byte
	if err := tx.QueryRow(ctx, `
		SELECT w.id,w.subject_id,w.witness_artifact_id,a.semantic_sha256,a.content_sha256,a.storage_key,
		       w.source_ref,w.source_sha256,w.metadata,w.superseded_by_product_revision_id,w.created_by,w.created_at
		FROM legacy_engineering_witnesses w
		JOIN artifacts a ON a.id=w.witness_artifact_id
		WHERE w.id=$1
	`, id).Scan(
		&out.ID,
		&out.SubjectID,
		&out.WitnessArtifactID,
		&out.ArtifactSemanticSHA256,
		&out.ArtifactContentSHA256,
		&out.ArtifactStorageKey,
		&out.SourceRef,
		&out.SourceSHA256,
		&metadata,
		&out.SupersededByProductRevisionID,
		&out.CreatedBy,
		&out.CreatedAt,
	); err != nil {
		return LegacyWitnessRecord{}, err
	}
	if err := json.Unmarshal(metadata, &out.Metadata); err != nil {
		return LegacyWitnessRecord{}, err
	}
	return out, nil
}

func activeLegacyWitness(ctx context.Context, pool *pgxpool.Pool, subjectID uuid.UUID) (LegacyWitnessRecord, error) {
	var id uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT id FROM legacy_engineering_witnesses
		WHERE subject_id=$1 AND superseded_by_product_revision_id IS NULL
	`, subjectID).Scan(&id); err != nil {
		return LegacyWitnessRecord{}, err
	}
	var out LegacyWitnessRecord
	var metadata []byte
	if err := pool.QueryRow(ctx, `
		SELECT w.id,w.subject_id,w.witness_artifact_id,a.semantic_sha256,a.content_sha256,a.storage_key,
		       w.source_ref,w.source_sha256,w.metadata,w.superseded_by_product_revision_id,w.created_by,w.created_at
		FROM legacy_engineering_witnesses w
		JOIN artifacts a ON a.id=w.witness_artifact_id
		WHERE w.id=$1
	`, id).Scan(
		&out.ID,
		&out.SubjectID,
		&out.WitnessArtifactID,
		&out.ArtifactSemanticSHA256,
		&out.ArtifactContentSHA256,
		&out.ArtifactStorageKey,
		&out.SourceRef,
		&out.SourceSHA256,
		&metadata,
		&out.SupersededByProductRevisionID,
		&out.CreatedBy,
		&out.CreatedAt,
	); err != nil {
		return LegacyWitnessRecord{}, err
	}
	if err := json.Unmarshal(metadata, &out.Metadata); err != nil {
		return LegacyWitnessRecord{}, err
	}
	return out, nil
}

func ResolveBaseline(ctx context.Context, pool *pgxpool.Pool, subjectSlug string) (Baseline, error) {
	subjectSlug = strings.TrimSpace(subjectSlug)
	if subjectSlug == "" {
		return Baseline{}, errors.New("subject slug is required")
	}
	var subjectID uuid.UUID
	var canonicalSlug string
	if err := pool.QueryRow(ctx, "SELECT id,slug FROM subjects WHERE slug=$1", subjectSlug).Scan(&subjectID, &canonicalSlug); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Baseline{}, ErrBaselineNotFound
		}
		return Baseline{}, err
	}

	products, err := loadCurrentProducts(ctx, pool, &subjectID)
	if err != nil {
		return Baseline{}, err
	}
	if len(products) > 1 {
		return Baseline{}, fmt.Errorf("SUBJECT_CURRENT_PRODUCT_CARDINALITY:%d", len(products))
	}
	if len(products) == 1 {
		product := products[0]
		return Baseline{
			Schema:      "RealSaS.SubjectBaseline.v1",
			SubjectID:   subjectID,
			SubjectSlug: canonicalSlug,
			Kind:        BaselineProductRevision,
			Product:     &product,
		}, nil
	}

	legacy, err := activeLegacyWitness(ctx, pool, subjectID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Baseline{}, ErrBaselineNotFound
		}
		return Baseline{}, err
	}
	return Baseline{
		Schema:        "RealSaS.SubjectBaseline.v1",
		SubjectID:     subjectID,
		SubjectSlug:   canonicalSlug,
		Kind:          BaselineLegacyEngineeringWitness,
		LegacyWitness: &legacy,
	}, nil
}
