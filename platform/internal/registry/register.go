package registry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
)

type ProducedObject struct {
	Role           string
	ArtifactType   string
	SchemaVersion  string
	StorageKey     string
	ContentSHA256  string
	SizeBytes      int64
	AuthorityClass string
}

type CapabilityIdentity struct {
	ID                   string
	ImplementationSHA256 string
	PolicySHA256         string
	ParametersSHA256     string
}

type inputIdentity struct {
	ID             uuid.UUID
	Role           string
	ArtifactType   string
	SemanticSHA256 string
}

type Registrar struct {
	Store artifactstore.Store
}

func (r Registrar) VerifyOutputs(ctx context.Context, outputs []ProducedObject) error {
	if r.Store == nil {
		return errors.New("artifact store is required")
	}
	seen := map[string]struct{}{}
	for _, output := range outputs {
		if output.Role == "" || output.ArtifactType == "" || output.SchemaVersion == "" {
			return errors.New("engine output role/type/schema are required")
		}
		if _, ok := seen[output.Role]; ok {
			return fmt.Errorf("duplicate engine output role %s", output.Role)
		}
		seen[output.Role] = struct{}{}
		if output.SizeBytes < 0 {
			return fmt.Errorf("negative output size for %s", output.Role)
		}
		if !strings.HasSuffix(output.StorageKey, output.ContentSHA256) {
			return fmt.Errorf("output storage key is not content-addressed for %s", output.Role)
		}
		if err := r.Store.Verify(ctx, artifactstore.Object{
			ContentSHA256: output.ContentSHA256,
			StorageKey:    output.StorageKey,
			SizeBytes:     output.SizeBytes,
		}); err != nil {
			return fmt.Errorf("verify output %s: %w", output.Role, err)
		}
	}
	return nil
}

func (r Registrar) RegisterCapabilityOutputs(
	ctx context.Context,
	tx pgx.Tx,
	executionID uuid.UUID,
	attemptID uuid.UUID,
	capability CapabilityIdentity,
	outputs []ProducedObject,
) ([]uuid.UUID, error) {
	if tx == nil {
		return nil, errors.New("transaction is required")
	}
	if capability.ID == "" {
		return nil, errors.New("capability id is required")
	}

	inputs, err := loadExecutionInputs(ctx, tx, executionID)
	if err != nil {
		return nil, err
	}
	semanticInputs := make([]domain.ArtifactInputIdentity, 0, len(inputs))
	for ordinal, input := range inputs {
		semanticInputs = append(semanticInputs, domain.ArtifactInputIdentity{
			Role:           input.Role,
			Ordinal:        ordinal,
			ArtifactType:   input.ArtifactType,
			SemanticSHA256: input.SemanticSHA256,
		})
	}

	artifactIDs := make([]uuid.UUID, 0, len(outputs))
	for _, output := range outputs {
		typeID, err := ensureArtifactType(ctx, tx, output.ArtifactType, output.SchemaVersion)
		if err != nil {
			return nil, err
		}
		descriptor := domain.ArtifactSemanticDescriptor{
			ArtifactType:         output.ArtifactType,
			SchemaVersion:        output.SchemaVersion,
			ProducerContract:     capability.ID,
			ImplementationSHA256: capability.ImplementationSHA256,
			PolicySHA256:         capability.PolicySHA256,
			Inputs:               semanticInputs,
			SemanticParameters: map[string]any{
				"capability_id":                capability.ID,
				"capability_parameters_sha256": capability.ParametersSHA256,
				"output_role":                  output.Role,
				"authority_class":              output.AuthorityClass,
			},
		}
		semanticSHA, err := descriptor.SemanticSHA256()
		if err != nil {
			return nil, err
		}
		paramsJSON, err := json.Marshal(descriptor.SemanticParameters)
		if err != nil {
			return nil, err
		}

		var artifactID uuid.UUID
		var existingContent, existingStorage string
		var existingSize int64
		err = tx.QueryRow(ctx, `
			SELECT id,content_sha256,storage_key,size_bytes
			FROM artifacts
			WHERE artifact_type_id=$1 AND semantic_sha256=$2
			FOR SHARE
		`, typeID, semanticSHA).Scan(&artifactID, &existingContent, &existingStorage, &existingSize)
		switch {
		case err == nil:
			if existingContent != output.ContentSHA256 || existingStorage != output.StorageKey || existingSize != output.SizeBytes {
				return nil, fmt.Errorf("ARTIFACT_NONDETERMINISM:%s:%s", capability.ID, output.Role)
			}
		case errors.Is(err, pgx.ErrNoRows):
			artifactID = uuid.New()
			if _, err := tx.Exec(ctx, `
				INSERT INTO artifacts
				  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				   producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now())
			`,
				artifactID,
				typeID,
				semanticSHA,
				output.ContentSHA256,
				output.StorageKey,
				output.SizeBytes,
				capability.ID,
				capability.ImplementationSHA256,
				capability.PolicySHA256,
				paramsJSON,
			); err != nil {
				return nil, err
			}
			for ordinal, input := range inputs {
				if _, err := tx.Exec(ctx, `
					INSERT INTO artifact_inputs(artifact_id,input_role,ordinal,input_artifact_id)
					VALUES ($1,$2,$3,$4)
				`, artifactID, input.Role, ordinal, input.ID); err != nil {
					return nil, err
				}
			}
		default:
			return nil, err
		}

		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'output',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, output.Role, artifactID); err != nil {
			return nil, err
		}
		attemptRole := "capability:" + capability.ID + ":" + output.Role
		tag, err := tx.Exec(ctx, `
			INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
			VALUES ($1,$2,$3,'produced')
			ON CONFLICT (attempt_id,role) DO NOTHING
		`, attemptID, attemptRole, artifactID)
		if err != nil {
			return nil, err
		}
		if tag.RowsAffected() == 0 {
			var existing uuid.UUID
			if err := tx.QueryRow(ctx, `
				SELECT artifact_id FROM attempt_artifacts WHERE attempt_id=$1 AND role=$2
			`, attemptID, attemptRole).Scan(&existing); err != nil {
				return nil, err
			}
			if existing != artifactID {
				return nil, fmt.Errorf("ATTEMPT_CAPABILITY_OUTPUT_DRIFT:%s:%s", capability.ID, output.Role)
			}
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO qualifications(id,artifact_id,qualification_type,result)
			VALUES ($1,$2,'CAPABILITY_EXECUTION_PASS','PASS')
			ON CONFLICT DO NOTHING
		`, uuid.New(), artifactID); err != nil {
			return nil, err
		}
		artifactIDs = append(artifactIDs, artifactID)
	}
	return artifactIDs, nil
}

func ensureArtifactType(ctx context.Context, tx pgx.Tx, name, schemaVersion string) (uuid.UUID, error) {
	id := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:artifact-type:"+name+":"+schemaVersion))
	if _, err := tx.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,$3,'capability')
		ON CONFLICT (name,schema_version) DO NOTHING
	`, id, name, schemaVersion); err != nil {
		return uuid.Nil, err
	}
	var actual uuid.UUID
	if err := tx.QueryRow(ctx, `
		SELECT id FROM artifact_types WHERE name=$1 AND schema_version=$2
	`, name, schemaVersion).Scan(&actual); err != nil {
		return uuid.Nil, err
	}
	return actual, nil
}

func loadExecutionInputs(ctx context.Context, tx pgx.Tx, executionID uuid.UUID) ([]inputIdentity, error) {
	rows, err := tx.Query(ctx, `
		SELECT a.id,ea.role,t.name,a.semantic_sha256
		FROM execution_artifacts ea
		JOIN artifacts a ON a.id=ea.artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE ea.execution_id=$1 AND ea.relation='input'
		ORDER BY ea.role,a.id
	`, executionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []inputIdentity
	for rows.Next() {
		var row inputIdentity
		if err := rows.Scan(&row.ID, &row.Role, &row.ArtifactType, &row.SemanticSHA256); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, rows.Err()
}
