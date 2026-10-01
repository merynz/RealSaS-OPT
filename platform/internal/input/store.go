package input

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
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type Binding struct {
	Role       string
	ArtifactID uuid.UUID
}

type ArtifactIdentity struct {
	Role                   string `json:"role"`
	Ordinal                int    `json:"ordinal"`
	ArtifactType           string `json:"artifact_type"`
	SchemaVersion          string `json:"schema_version"`
	ArtifactSemanticSHA256 string `json:"artifact_semantic_sha256"`
}

type Manifest struct {
	ContractVersion string             `json:"contract_version"`
	SubjectID       string             `json:"subject_id"`
	Artifacts       []ArtifactIdentity `json:"artifacts"`
}

type Sealed struct {
	SubjectInputID uuid.UUID
	ManifestSHA256 string
	Reused         bool
}

type Loaded struct {
	SubjectInputID uuid.UUID
	SubjectID      uuid.UUID
	ManifestSHA256 string
	RootInputs     []domain.ArtifactInputIdentity
	ArtifactIDs    []uuid.UUID
}

var (
	ErrSubjectNotFound  = errors.New("subject input subject not found")
	ErrArtifactNotFound = errors.New("subject input artifact not found")
	ErrSubjectMismatch  = errors.New("subject input subject mismatch")
	ErrEmpty            = errors.New("subject input is empty")
)

func Seal(ctx context.Context, pool *pgxpool.Pool, subjectID uuid.UUID, bindings []Binding, createdBy string) (Sealed, error) {
	if subjectID == uuid.Nil || len(bindings) == 0 || createdBy == "" {
		return Sealed{}, errors.New("subject, bindings and created_by are required")
	}
	seenRoles := map[string]struct{}{}
	for _, b := range bindings {
		if b.Role == "" || b.ArtifactID == uuid.Nil {
			return Sealed{}, errors.New("subject input bindings require role and artifact id")
		}
		if _, ok := seenRoles[b.Role]; ok {
			return Sealed{}, fmt.Errorf("duplicate subject input role %q", b.Role)
		}
		seenRoles[b.Role] = struct{}{}
	}

	var out Sealed
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var subject uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", subjectID).Scan(&subject); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrSubjectNotFound
			}
			return err
		}

		identities := make([]ArtifactIdentity, 0, len(bindings))
		artifactIDs := make([]uuid.UUID, 0, len(bindings))
		for ordinal, binding := range bindings {
			var artifactID uuid.UUID
			var artifactType, schemaVersion, semanticSHA string
			err := tx.QueryRow(ctx, `
				SELECT a.id,t.name,t.schema_version,a.semantic_sha256
				FROM artifacts a
				JOIN artifact_types t ON t.id=a.artifact_type_id
				WHERE a.id=$1
			`, binding.ArtifactID).Scan(&artifactID, &artifactType, &schemaVersion, &semanticSHA)
			if err != nil {
				if errors.Is(err, pgx.ErrNoRows) {
					return fmt.Errorf("%w: %s", ErrArtifactNotFound, binding.ArtifactID)
				}
				return err
			}
			identities = append(identities, ArtifactIdentity{
				Role:                   binding.Role,
				Ordinal:                ordinal,
				ArtifactType:           artifactType,
				SchemaVersion:          schemaVersion,
				ArtifactSemanticSHA256: semanticSHA,
			})
			artifactIDs = append(artifactIDs, artifactID)
		}

		manifest := Manifest{
			ContractVersion: "RealSaS.SubjectInputManifest.v1",
			SubjectID:       subjectID.String(),
			Artifacts:       identities,
		}
		manifestSHA, err := semantic.JSONSHA256(manifest)
		if err != nil {
			return err
		}

		var existing uuid.UUID
		err = tx.QueryRow(ctx, `
			SELECT id FROM subject_inputs
			WHERE subject_id=$1 AND manifest_sha256=$2
		`, subjectID, manifestSHA).Scan(&existing)
		switch {
		case err == nil:
			out = Sealed{SubjectInputID: existing, ManifestSHA256: manifestSHA, Reused: true}
			return nil
		case !errors.Is(err, pgx.ErrNoRows):
			return err
		}

		id := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO subject_inputs(id,subject_id,manifest_sha256,created_by,sealed_at)
			VALUES ($1,$2,$3,$4,now())
		`, id, subjectID, manifestSHA, createdBy); err != nil {
			return err
		}
		for i, identity := range identities {
			if _, err := tx.Exec(ctx, `
				INSERT INTO subject_input_artifacts(subject_input_id,role,ordinal,artifact_id)
				VALUES ($1,$2,$3,$4)
			`, id, identity.Role, identity.Ordinal, artifactIDs[i]); err != nil {
				return err
			}
		}
		audit, _ := json.Marshal(map[string]any{
			"subject_input_id": id.String(),
			"manifest_sha256":  manifestSHA,
			"roles":            seenRolesSlice(identities),
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,payload)
			VALUES ($1,'SUBJECT_INPUT_SEALED',$2,$3)
		`, createdBy, subjectID, audit); err != nil {
			return err
		}
		out = Sealed{SubjectInputID: id, ManifestSHA256: manifestSHA, Reused: false}
		return nil
	})
	return out, err
}

func Load(ctx context.Context, pool *pgxpool.Pool, subjectInputID, subjectID uuid.UUID) (Loaded, error) {
	var actualSubject uuid.UUID
	var manifestSHA string
	if err := pool.QueryRow(ctx, `
		SELECT subject_id,manifest_sha256 FROM subject_inputs WHERE id=$1
	`, subjectInputID).Scan(&actualSubject, &manifestSHA); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Loaded{}, ErrEmpty
		}
		return Loaded{}, err
	}
	if actualSubject != subjectID {
		return Loaded{}, ErrSubjectMismatch
	}
	rows, err := pool.Query(ctx, `
		SELECT sia.role,sia.ordinal,sia.artifact_id,t.name,a.semantic_sha256
		FROM subject_input_artifacts sia
		JOIN artifacts a ON a.id=sia.artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE sia.subject_input_id=$1
		ORDER BY sia.ordinal
	`, subjectInputID)
	if err != nil {
		return Loaded{}, err
	}
	defer rows.Close()

	var roots []domain.ArtifactInputIdentity
	var artifactIDs []uuid.UUID
	for rows.Next() {
		var role, artifactType, semanticSHA string
		var ordinal int
		var artifactID uuid.UUID
		if err := rows.Scan(&role, &ordinal, &artifactID, &artifactType, &semanticSHA); err != nil {
			return Loaded{}, err
		}
		roots = append(roots, domain.ArtifactInputIdentity{
			Role:           "subject:" + role,
			Ordinal:        ordinal,
			ArtifactType:   artifactType,
			SemanticSHA256: semanticSHA,
		})
		artifactIDs = append(artifactIDs, artifactID)
	}
	if err := rows.Err(); err != nil {
		return Loaded{}, err
	}
	if len(roots) == 0 {
		return Loaded{}, ErrEmpty
	}
	return Loaded{
		SubjectInputID: subjectInputID,
		SubjectID:      subjectID,
		ManifestSHA256: manifestSHA,
		RootInputs:     roots,
		ArtifactIDs:    artifactIDs,
	}, nil
}

func seenRolesSlice(identities []ArtifactIdentity) []string {
	roles := make([]string, 0, len(identities))
	for _, item := range identities {
		roles = append(roles, item.Role)
	}
	return roles
}
