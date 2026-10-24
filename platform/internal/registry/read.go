package registry

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
)

var (
	ErrArtifactNotFound = errors.New("artifact not found")
	ErrSubjectNotFound  = errors.New("subject not found")
)

type ArtifactInputRecord struct {
	Role           string    `json:"role"`
	Ordinal        int       `json:"ordinal"`
	ArtifactID     uuid.UUID `json:"artifact_id"`
	ArtifactType   string    `json:"artifact_type"`
	SchemaVersion  string    `json:"schema_version"`
	SemanticSHA256 string    `json:"semantic_sha256"`
}

type ArtifactQualificationRecord struct {
	QualificationType string     `json:"qualification_type"`
	Result            string     `json:"result"`
	ProofID           *uuid.UUID `json:"proof_id,omitempty"`
	CreatedAt         time.Time  `json:"created_at"`
}

type ArtifactRecord struct {
	ID                   uuid.UUID                     `json:"id"`
	ArtifactType         string                        `json:"artifact_type"`
	SchemaVersion        string                        `json:"schema_version"`
	Domain               string                        `json:"domain"`
	SemanticSHA256       string                        `json:"semantic_sha256"`
	ContentSHA256        string                        `json:"content_sha256"`
	StorageKey           string                        `json:"storage_key"`
	SizeBytes            int64                         `json:"size_bytes"`
	ProducerContract     string                        `json:"producer_contract"`
	ImplementationSHA256 string                        `json:"implementation_sha256"`
	PolicySHA256         string                        `json:"policy_sha256"`
	SemanticParameters   map[string]any                `json:"semantic_parameters"`
	CreatedAt            time.Time                     `json:"created_at"`
	VerifiedAt           time.Time                     `json:"verified_at"`
	Inputs               []ArtifactInputRecord         `json:"inputs"`
	Qualifications       []ArtifactQualificationRecord `json:"qualifications"`
}

type SubjectArtifactRecord struct {
	SourceKind           string    `json:"source_kind"`
	SourceID             uuid.UUID `json:"source_id"`
	Role                 string    `json:"role"`
	Ordinal              int       `json:"ordinal"`
	Origin               string    `json:"origin"`
	ObservedAt           time.Time `json:"observed_at"`
	ArtifactID           uuid.UUID `json:"artifact_id"`
	ArtifactType         string    `json:"artifact_type"`
	SchemaVersion        string    `json:"schema_version"`
	SemanticSHA256       string    `json:"semantic_sha256"`
	ContentSHA256        string    `json:"content_sha256"`
	SizeBytes            int64     `json:"size_bytes"`
	ProducerContract     string    `json:"producer_contract"`
	ImplementationSHA256 string    `json:"implementation_sha256"`
	PolicySHA256         string    `json:"policy_sha256"`
}

func GetArtifact(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (ArtifactRecord, error) {
	if pool == nil {
		return ArtifactRecord{}, errors.New("artifact registry database is required")
	}
	if id == uuid.Nil {
		return ArtifactRecord{}, ErrArtifactNotFound
	}

	var out ArtifactRecord
	var params []byte
	err := pool.QueryRow(ctx, `
		SELECT a.id,t.name,t.schema_version,t.domain,a.semantic_sha256,a.content_sha256,
		       a.storage_key,a.size_bytes,a.producer_contract,a.implementation_sha256,
		       a.policy_sha256,a.semantic_parameters,a.created_at,a.verified_at
		FROM artifacts a
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE a.id=$1
	`, id).Scan(
		&out.ID,
		&out.ArtifactType,
		&out.SchemaVersion,
		&out.Domain,
		&out.SemanticSHA256,
		&out.ContentSHA256,
		&out.StorageKey,
		&out.SizeBytes,
		&out.ProducerContract,
		&out.ImplementationSHA256,
		&out.PolicySHA256,
		&params,
		&out.CreatedAt,
		&out.VerifiedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ArtifactRecord{}, ErrArtifactNotFound
	}
	if err != nil {
		return ArtifactRecord{}, err
	}
	if err := json.Unmarshal(params, &out.SemanticParameters); err != nil {
		return ArtifactRecord{}, fmt.Errorf("artifact semantic parameters: %w", err)
	}
	if out.SemanticParameters == nil {
		out.SemanticParameters = map[string]any{}
	}

	inputRows, err := pool.Query(ctx, `
		SELECT ai.input_role,ai.ordinal,a.id,t.name,t.schema_version,a.semantic_sha256
		FROM artifact_inputs ai
		JOIN artifacts a ON a.id=ai.input_artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE ai.artifact_id=$1
		ORDER BY ai.ordinal,ai.input_role,a.id
	`, id)
	if err != nil {
		return ArtifactRecord{}, err
	}
	for inputRows.Next() {
		var row ArtifactInputRecord
		if err := inputRows.Scan(&row.Role, &row.Ordinal, &row.ArtifactID, &row.ArtifactType, &row.SchemaVersion, &row.SemanticSHA256); err != nil {
			inputRows.Close()
			return ArtifactRecord{}, err
		}
		out.Inputs = append(out.Inputs, row)
	}
	if err := inputRows.Err(); err != nil {
		inputRows.Close()
		return ArtifactRecord{}, err
	}
	inputRows.Close()
	if out.Inputs == nil {
		out.Inputs = []ArtifactInputRecord{}
	}

	qualificationRows, err := pool.Query(ctx, `
		SELECT qualification_type,result,proof_id,created_at
		FROM qualifications
		WHERE artifact_id=$1
		ORDER BY qualification_type,created_at,id
	`, id)
	if err != nil {
		return ArtifactRecord{}, err
	}
	for qualificationRows.Next() {
		var row ArtifactQualificationRecord
		if err := qualificationRows.Scan(&row.QualificationType, &row.Result, &row.ProofID, &row.CreatedAt); err != nil {
			qualificationRows.Close()
			return ArtifactRecord{}, err
		}
		out.Qualifications = append(out.Qualifications, row)
	}
	if err := qualificationRows.Err(); err != nil {
		qualificationRows.Close()
		return ArtifactRecord{}, err
	}
	qualificationRows.Close()
	if out.Qualifications == nil {
		out.Qualifications = []ArtifactQualificationRecord{}
	}
	return out, nil
}

func ListSubjectArtifacts(
	ctx context.Context,
	pool *pgxpool.Pool,
	subjectID uuid.UUID,
	artifactType string,
	schemaVersion string,
	limit int,
) ([]SubjectArtifactRecord, error) {
	if pool == nil {
		return nil, errors.New("artifact registry database is required")
	}
	if subjectID == uuid.Nil {
		return nil, ErrSubjectNotFound
	}
	artifactType = strings.TrimSpace(artifactType)
	schemaVersion = strings.TrimSpace(schemaVersion)
	if limit <= 0 || limit > 500 {
		return nil, errors.New("artifact list limit must be between 1 and 500")
	}

	var exists bool
	if err := pool.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM subjects WHERE id=$1)", subjectID).Scan(&exists); err != nil {
		return nil, err
	}
	if !exists {
		return nil, ErrSubjectNotFound
	}

	rows, err := pool.Query(ctx, `
		WITH bindings AS (
			SELECT 'SUBJECT_INPUT'::text AS source_kind,si.id AS source_id,sia.role,
			       sia.ordinal,'sealed'::text AS origin,si.sealed_at AS observed_at,sia.artifact_id
			FROM subject_inputs si
			JOIN subject_input_artifacts sia ON sia.subject_input_id=si.id
			WHERE si.subject_id=$1
			UNION ALL
			SELECT 'ATTEMPT'::text,at.id,aa.role,0,aa.origin,at.created_at,aa.artifact_id
			FROM attempts at
			JOIN attempt_artifacts aa ON aa.attempt_id=at.id
			WHERE at.subject_id=$1
			UNION ALL
			SELECT 'PRODUCT_REVISION'::text,pr.id,pra.role,0,
			       CASE WHEN scr.product_revision_id IS NOT NULL THEN 'current' ELSE 'sealed' END,
			       pr.sealed_at,pra.artifact_id
			FROM product_revisions pr
			JOIN product_revision_artifacts pra ON pra.product_revision_id=pr.id
			LEFT JOIN subject_current_revision scr
			  ON scr.subject_id=pr.subject_id AND scr.product_revision_id=pr.id
			WHERE pr.subject_id=$1
		)
		SELECT b.source_kind,b.source_id,b.role,b.ordinal,b.origin,b.observed_at,
		       a.id,t.name,t.schema_version,a.semantic_sha256,a.content_sha256,a.size_bytes,
		       a.producer_contract,a.implementation_sha256,a.policy_sha256
		FROM bindings b
		JOIN artifacts a ON a.id=b.artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE ($2='' OR t.name=$2)
		  AND ($3='' OR t.schema_version=$3)
		ORDER BY b.observed_at DESC,b.source_kind,b.source_id,b.role,b.ordinal
		LIMIT $4
	`, subjectID, artifactType, schemaVersion, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]SubjectArtifactRecord, 0)
	for rows.Next() {
		var row SubjectArtifactRecord
		if err := rows.Scan(
			&row.SourceKind,
			&row.SourceID,
			&row.Role,
			&row.Ordinal,
			&row.Origin,
			&row.ObservedAt,
			&row.ArtifactID,
			&row.ArtifactType,
			&row.SchemaVersion,
			&row.SemanticSHA256,
			&row.ContentSHA256,
			&row.SizeBytes,
			&row.ProducerContract,
			&row.ImplementationSHA256,
			&row.PolicySHA256,
		); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, rows.Err()
}
