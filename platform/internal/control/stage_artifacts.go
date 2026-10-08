package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/resolver"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type stageResultManifest struct {
	SemanticParameters     map[string]any               `json:"semantic_parameters"`
	ImplementationSHA256   string                       `json:"implementation_sha256"`
	PolicySHA256           string                       `json:"policy_sha256"`
	Schema                 string                       `json:"schema"`
	StageID                string                       `json:"stage_id"`
	CompilerStatus         string                       `json:"compiler_status"`
	ExpectedSemanticSHA256 string                       `json:"expected_semantic_sha256"`
	DiagnosticsHash        string                       `json:"diagnostics_hash"`
	Outputs                []orchestration.EngineOutput `json:"outputs"`
}

type preparedStageResultArtifact struct {
	Object                 artifactstore.Object
	Manifest               stageResultManifest
	ImplementationSHA256   string
	PolicySHA256           string
	SemanticParametersJSON []byte
}

func (a Activities) bindStageExecutionInputs(
	ctx context.Context,
	tx pgx.Tx,
	executionID uuid.UUID,
	subjectInputID string,
	attemptID uuid.UUID,
	stageID string,
) error {
	inputID, err := uuid.Parse(subjectInputID)
	if err != nil {
		return errors.New("STAGE_EXECUTION_SUBJECT_INPUT_REQUIRED")
	}
	rows, err := tx.Query(ctx, `
		SELECT sia.role,sia.artifact_id
		FROM subject_input_artifacts sia
		WHERE sia.subject_input_id=$1
		ORDER BY sia.ordinal
	`, inputID)
	if err != nil {
		return err
	}
	type subjectInputBinding struct {
		role       string
		artifactID uuid.UUID
	}
	var subjectInputs []subjectInputBinding
	for rows.Next() {
		var item subjectInputBinding
		if err := rows.Scan(&item.role, &item.artifactID); err != nil {
			rows.Close()
			return err
		}
		subjectInputs = append(subjectInputs, item)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()
	stage, ok := a.Graph.Get(stageID)
	if !ok {
		return fmt.Errorf("unknown stage %s", stageID)
	}
	for _, item := range subjectInputs {
		if !stage.ConsumesInput(item.role) {
			continue
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'input',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, "subject:"+item.role, item.artifactID); err != nil {
			return err
		}
	}

	for _, dependency := range stage.DependsOn {
		var artifactID uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT artifact_id
			FROM attempt_artifacts
			WHERE attempt_id=$1 AND role=$2
		`, attemptID, "stage:"+dependency).Scan(&artifactID); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return fmt.Errorf("STAGE_DEPENDENCY_ARTIFACT_REQUIRED:%s:%s", stageID, dependency)
			}
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
			VALUES ($1,'input',$2,$3)
			ON CONFLICT DO NOTHING
		`, executionID, "stage:"+dependency, artifactID); err != nil {
			return err
		}
	}
	return nil
}

func (a Activities) prepareStageResultArtifact(
	ctx context.Context,
	attemptID uuid.UUID,
	req orchestration.StageCommitRequest,
) (*preparedStageResultArtifact, error) {
	if a.Store == nil {
		return nil, errors.New("stage result commit requires artifact store")
	}
	if len(req.EngineResult.Outputs) == 0 {
		return nil, errors.New("PASS_STAGE_REQUIRES_OUTPUTS")
	}
	outputs := append([]orchestration.EngineOutput(nil), req.EngineResult.Outputs...)
	sort.Slice(outputs, func(i, j int) bool {
		if outputs[i].Role != outputs[j].Role {
			return outputs[i].Role < outputs[j].Role
		}
		if outputs[i].AuthorityClass != outputs[j].AuthorityClass {
			return outputs[i].AuthorityClass < outputs[j].AuthorityClass
		}
		return outputs[i].StorageKey < outputs[j].StorageKey
	})
	seenRoles := map[string]struct{}{}
	for _, output := range outputs {
		if output.RelativePath == "" || path.IsAbs(output.RelativePath) || path.Clean(output.RelativePath) != output.RelativePath || output.RelativePath == ".." || strings.HasPrefix(output.RelativePath, "../") || output.PayloadSchema == "" {
			return nil, errors.New("ENGINE_STAGE_PORTABLE_OUTPUT_REQUIRED")
		}
		if output.Role == "" || output.ArtifactType == "" || output.SchemaVersion == "" {
			return nil, errors.New("ENGINE_STAGE_OUTPUT_IDENTITY_INCOMPLETE")
		}
		if _, exists := seenRoles[output.Role]; exists {
			return nil, fmt.Errorf("ENGINE_STAGE_OUTPUT_ROLE_DUPLICATE:%s", output.Role)
		}
		seenRoles[output.Role] = struct{}{}
		if err := a.Store.Verify(ctx, artifactstore.Object{
			ContentSHA256: output.ContentSHA256,
			StorageKey:    output.StorageKey,
			SizeBytes:     output.SizeBytes,
		}); err != nil {
			return nil, fmt.Errorf("ENGINE_STAGE_OUTPUT_VERIFY:%s:%w", output.Role, err)
		}
	}
	var implementationSHA, policySHA string
	var parametersRaw []byte
	if err := a.Pool.QueryRow(ctx, `
		SELECT ers.implementation_sha256,ers.policy_sha256,ers.semantic_parameters
		FROM attempts a
		JOIN engine_release_stages ers ON ers.release_id=a.engine_release_id
		WHERE a.id=$1 AND ers.stage_id=$2
	`, attemptID, req.StageID).Scan(&implementationSHA, &policySHA, &parametersRaw); err != nil {
		return nil, err
	}
	var parameters map[string]any
	if err := json.Unmarshal(parametersRaw, &parameters); err != nil {
		return nil, err
	}
	manifest := stageResultManifest{
		SemanticParameters:   parameters,
		ImplementationSHA256: implementationSHA, PolicySHA256: policySHA,
		Schema:                 "RealSaS.StageResultManifest.v1",
		StageID:                req.StageID,
		CompilerStatus:         req.EngineResult.Status,
		ExpectedSemanticSHA256: req.ExpectedSemanticSHA256,
		DiagnosticsHash:        req.EngineResult.DiagnosticsHash,
		Outputs:                outputs,
	}
	raw, err := semantic.CanonicalJSON(manifest)
	if err != nil {
		return nil, err
	}
	object, err := a.Store.PutBytes(ctx, raw)
	if err != nil {
		return nil, err
	}
	params, err := json.Marshal(map[string]any{
		"stage_id":        req.StageID,
		"compiler_status": req.EngineResult.Status,
		"output_count":    len(outputs),
	})
	if err != nil {
		return nil, err
	}
	return &preparedStageResultArtifact{
		Object:                 object,
		Manifest:               manifest,
		ImplementationSHA256:   implementationSHA,
		PolicySHA256:           policySHA,
		SemanticParametersJSON: params,
	}, nil
}

func (a Activities) commitStageResultArtifact(
	ctx context.Context,
	tx pgx.Tx,
	executionID uuid.UUID,
	attemptID uuid.UUID,
	req orchestration.StageCommitRequest,
	prepared *preparedStageResultArtifact,
) (uuid.UUID, error) {
	if prepared == nil {
		return uuid.Nil, errors.New("prepared stage result artifact required")
	}
	typeID := uuid.NewSHA1(
		uuid.NameSpaceURL,
		[]byte("realsas:artifact-type:"+resolver.StageResultArtifactType+":"+resolver.StageResultSchema),
	)
	if _, err := tx.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,$3,'stage')
		ON CONFLICT (name,schema_version) DO NOTHING
	`, typeID, resolver.StageResultArtifactType, resolver.StageResultSchema); err != nil {
		return uuid.Nil, err
	}
	if err := tx.QueryRow(ctx, `
		SELECT id FROM artifact_types WHERE name=$1 AND schema_version=$2
	`, resolver.StageResultArtifactType, resolver.StageResultSchema).Scan(&typeID); err != nil {
		return uuid.Nil, err
	}

	var artifactID uuid.UUID
	var existingContent, existingStorage string
	var existingSize int64
	err := tx.QueryRow(ctx, `
		SELECT id,content_sha256,storage_key,size_bytes
		FROM artifacts
		WHERE artifact_type_id=$1 AND semantic_sha256=$2
		FOR SHARE
	`, typeID, req.ExpectedSemanticSHA256).Scan(&artifactID, &existingContent, &existingStorage, &existingSize)
	switch {
	case err == nil:
		if existingContent != prepared.Object.ContentSHA256 ||
			existingStorage != prepared.Object.StorageKey ||
			existingSize != prepared.Object.SizeBytes {
			return uuid.Nil, fmt.Errorf("ARTIFACT_NONDETERMINISM:%s", req.StageID)
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
			req.ExpectedSemanticSHA256,
			prepared.Object.ContentSHA256,
			prepared.Object.StorageKey,
			prepared.Object.SizeBytes,
			req.StageID,
			prepared.ImplementationSHA256,
			prepared.PolicySHA256,
			prepared.SemanticParametersJSON,
		); err != nil {
			return uuid.Nil, err
		}
		rows, err := tx.Query(ctx, `
			SELECT role,artifact_id
			FROM execution_artifacts
			WHERE execution_id=$1 AND relation='input'
			ORDER BY role,artifact_id
		`, executionID)
		if err != nil {
			return uuid.Nil, err
		}
		type dependency struct {
			role string
			id   uuid.UUID
		}
		var dependencies []dependency
		for rows.Next() {
			var role string
			var inputID uuid.UUID
			if err := rows.Scan(&role, &inputID); err != nil {
				rows.Close()
				return uuid.Nil, err
			}
			dependencies = append(dependencies, dependency{role: role, id: inputID})
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return uuid.Nil, err
		}
		rows.Close()
		// pgx cannot issue INSERTs while a result set still owns the connection.
		for ordinal, input := range dependencies {
			if _, err := tx.Exec(ctx, `INSERT INTO artifact_inputs
				(artifact_id,input_role,ordinal,input_artifact_id) VALUES ($1,$2,$3,$4)`,
				artifactID, input.role, ordinal, input.id); err != nil {
				return uuid.Nil, err
			}
		}
	default:
		return uuid.Nil, err
	}

	if _, err := tx.Exec(ctx, `
		INSERT INTO execution_artifacts(execution_id,relation,role,artifact_id)
		VALUES ($1,'output','stage-result',$2)
		ON CONFLICT DO NOTHING
	`, executionID, artifactID); err != nil {
		return uuid.Nil, err
	}
	role := "stage:" + req.StageID
	tag, err := tx.Exec(ctx, `
		INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
		VALUES ($1,$2,$3,'produced')
		ON CONFLICT (attempt_id,role) DO NOTHING
	`, attemptID, role, artifactID)
	if err != nil {
		return uuid.Nil, err
	}
	if tag.RowsAffected() == 0 {
		var existing uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT artifact_id FROM attempt_artifacts WHERE attempt_id=$1 AND role=$2
		`, attemptID, role).Scan(&existing); err != nil {
			return uuid.Nil, err
		}
		if existing != artifactID {
			return uuid.Nil, fmt.Errorf("ATTEMPT_STAGE_OUTPUT_DRIFT:%s", req.StageID)
		}
	}
	qualification := "REUSE_ELIGIBLE"
	if req.EngineResult.Status == "PASS_DEMO_ONLY" {
		qualification = "DEMO_REUSE_ELIGIBLE"
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO qualifications(id,artifact_id,qualification_type,result)
		VALUES ($1,$2,$3,'PASS')
		ON CONFLICT DO NOTHING
	`, uuid.New(), artifactID, qualification); err != nil {
		return uuid.Nil, err
	}
	return artifactID, nil
}
