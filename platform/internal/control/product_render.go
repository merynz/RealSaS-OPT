package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/product"
	"github.com/merynz/RealSaS-OPT/platform/internal/resolver"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

const (
	productMotionRole  = "motion:qualified"
	productRuntimeRole = "runtime:package"

	motionArtifactType  = "RealSaS.QualifiedMotionArtifact"
	runtimeArtifactType = "RealSaS.RuntimePackageArtifact"
	renderArtifactType  = "RealSaS.RenderOutput"
	platformSchemaV1    = "v1"
)

type finalizeCompilePayload struct {
	Command         orchestration.CompileWorkflowInput `json:"command"`
	Plan            orchestration.ResolvedCompilePlan  `json:"plan"`
	CompletedStages []orchestration.StageCommitResult  `json:"completed_stages"`
}

type revisionBinding struct {
	Role           string `json:"role"`
	ArtifactID     string `json:"artifact_id"`
	ArtifactType   string `json:"artifact_type"`
	SchemaVersion  string `json:"schema_version"`
	SemanticSHA256 string `json:"semantic_sha256"`
}

func (a Activities) FinalizeCompile(ctx context.Context, payload map[string]any) (orchestration.CompileWorkflowResult, error) {
	if err := a.Validate(); err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	if a.Store == nil {
		return orchestration.CompileWorkflowResult{}, errors.New("finalize compile requires artifact store")
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	var in finalizeCompilePayload
	if err := json.Unmarshal(raw, &in); err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	attemptID, err := uuid.Parse(in.Command.AttemptID)
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	subjectID, err := uuid.Parse(in.Command.SubjectID)
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	for _, row := range in.CompletedStages {
		if row.Status == "FAIL" {
			return orchestration.CompileWorkflowResult{}, errors.New("CANNOT_FINALIZE_FAILED_COMPILE")
		}
	}
	result := orchestration.CompileWorkflowResult{
		AttemptID:       attemptID.String(),
		Status:          "PASS",
		CompletedStages: append([]orchestration.StageCommitResult(nil), in.CompletedStages...),
	}
	if in.Plan.TargetStageID != "46_PRODUCT_CLOSURE_SEAL" {
		_, err := a.Pool.Exec(ctx, "UPDATE attempts SET final_state='COMPLETED' WHERE id=$1 AND final_state='OPEN'", attemptID)
		return result, err
	}

	motionID, err := a.promoteStageOutput(
		ctx,
		attemptID,
		"40_MOTION_COMPILE_RUN",
		"QUALIFIED_FULL_3D_MOTION_V2",
		productMotionRole,
		motionArtifactType,
		platformSchemaV1,
		"motion",
		"MOTION_COMPILED",
	)
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	runtimeID, err := a.promoteStageOutput(
		ctx,
		attemptID,
		"43_RSS_MATERIALIZE_COMPACT",
		"RUNTIME_V2_RSS_PACKAGE",
		productRuntimeRole,
		runtimeArtifactType,
		platformSchemaV1,
		"runtime",
		"RUNTIME_PACKAGE_QUALIFIED",
	)
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}

	contract, err := product.SealContract(ctx, a.Pool, product.Contract{
		Name:    "RealSaS.ProductRevision",
		Version: "v1",
		Roles: []product.RoleRule{
			{
				Role:              "stage:46_PRODUCT_CLOSURE_SEAL",
				ArtifactType:      resolver.StageResultArtifactType,
				SchemaVersion:     resolver.StageResultSchema,
				QualificationType: "REUSE_ELIGIBLE",
				Required:          true,
			},
			{
				Role:              productMotionRole,
				ArtifactType:      motionArtifactType,
				SchemaVersion:     platformSchemaV1,
				QualificationType: "MOTION_COMPILED",
				Required:          true,
			},
			{
				Role:              productRuntimeRole,
				ArtifactType:      runtimeArtifactType,
				SchemaVersion:     platformSchemaV1,
				QualificationType: "RUNTIME_PACKAGE_QUALIFIED",
				Required:          true,
			},
		},
		Metadata: map[string]any{
			"compiler_product_pass_stage": "46_PRODUCT_CLOSURE_SEAL",
			"render_is_tail_only":         true,
		},
	}, "platform.finalize_compile_attempt.v1")
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}

	var revisionID uuid.UUID
	err = persistence.WithSerializableRetry(ctx, a.Pool, 5, func(tx pgx.Tx) error {
		var dbSubject uuid.UUID
		var releaseID *uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,engine_release_id
			FROM attempts
			WHERE id=$1
			FOR UPDATE
		`, attemptID).Scan(&dbSubject, &releaseID); err != nil {
			return err
		}
		if dbSubject != subjectID || releaseID == nil {
			return errors.New("FINALIZE_ATTEMPT_IDENTITY_DRIFT")
		}
		if _, err := tx.Exec(ctx, "SELECT id FROM subjects WHERE id=$1 FOR UPDATE", subjectID); err != nil {
			return err
		}

		rows, err := tx.Query(ctx, `
			SELECT aa.role,a.id,t.name,t.schema_version,a.semantic_sha256
			FROM attempt_artifacts aa
			JOIN artifacts a ON a.id=aa.artifact_id
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE aa.attempt_id=$1
			ORDER BY aa.role
		`, attemptID)
		if err != nil {
			return err
		}
		var bindings []revisionBinding
		for rows.Next() {
			var b revisionBinding
			if err := rows.Scan(&b.Role, &b.ArtifactID, &b.ArtifactType, &b.SchemaVersion, &b.SemanticSHA256); err != nil {
				rows.Close()
				return err
			}
			bindings = append(bindings, b)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return err
		}
		rows.Close()
		required := map[string]bool{
			"stage:46_PRODUCT_CLOSURE_SEAL": false,
			productMotionRole:                false,
			productRuntimeRole:               false,
		}
		for _, b := range bindings {
			if _, ok := required[b.Role]; ok {
				required[b.Role] = true
			}
		}
		if !required["stage:46_PRODUCT_CLOSURE_SEAL"] || !required[productMotionRole] || !required[productRuntimeRole] {
			return errors.New("PRODUCT_REVISION_REQUIRED_BINDINGS_MISSING")
		}
		manifest := map[string]any{
			"schema":                  "RealSaS.ProductRevisionManifest.v1",
			"subject_id":              subjectID.String(),
			"engine_release_id":       releaseID.String(),
			"product_contract_sha256": contract.ContractSHA256,
			"bindings":                bindings,
		}
		manifestSHA, err := semantic.JSONSHA256(manifest)
		if err != nil {
			return err
		}
		err = tx.QueryRow(ctx, `
			SELECT id
			FROM product_revisions
			WHERE subject_id=$1 AND manifest_sha256=$2
		`, subjectID, manifestSHA).Scan(&revisionID)
		if err == nil {
			if _, err := tx.Exec(ctx, "UPDATE attempts SET final_state='QUALIFIED' WHERE id=$1", attemptID); err != nil {
				return err
			}
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		var nextRevision int64
		if err := tx.QueryRow(ctx, `
			SELECT COALESCE(MAX(revision_number),0)+1
			FROM product_revisions
			WHERE subject_id=$1
		`, subjectID).Scan(&nextRevision); err != nil {
			return err
		}
		revisionID = uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO product_revisions
			  (id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at,product_contract_id)
			VALUES ($1,$2,$3,$4,$5,now(),$6)
		`, revisionID, subjectID, nextRevision, manifestSHA, attemptID, contract.ID); err != nil {
			return err
		}
		for _, b := range bindings {
			artifactID, err := uuid.Parse(b.ArtifactID)
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
				INSERT INTO product_revision_artifacts(product_revision_id,role,artifact_id)
				VALUES ($1,$2,$3)
			`, revisionID, b.Role, artifactID); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, "UPDATE attempts SET final_state='QUALIFIED' WHERE id=$1", attemptID); err != nil {
			return err
		}
		body, _ := json.Marshal(map[string]any{
			"product_revision_id": revisionID.String(),
			"manifest_sha256":     manifestSHA,
			"motion_artifact_id":  motionID.String(),
			"runtime_artifact_id": runtimeID.String(),
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'PRODUCT_REVISION_SEALED',$2)
		`, attemptID, body); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return orchestration.CompileWorkflowResult{}, err
	}
	value := revisionID.String()
	result.ProductRevisionID = &value
	return result, nil
}

func (a Activities) promoteStageOutput(
	ctx context.Context,
	attemptID uuid.UUID,
	stageID string,
	authorityClass string,
	productRole string,
	artifactType string,
	schemaVersion string,
	domain string,
	qualification string,
) (uuid.UUID, error) {
	if a.Store == nil {
		return uuid.Nil, errors.New("promote stage output requires artifact store")
	}
	var stageArtifactID uuid.UUID
	var stageSemantic, stageStorage, stageContent, implementationSHA, policySHA string
	var stageSize int64
	if err := a.Pool.QueryRow(ctx, `
		SELECT a.id,a.semantic_sha256,a.storage_key,a.content_sha256,a.size_bytes,
		       a.implementation_sha256,a.policy_sha256
		FROM attempt_artifacts aa
		JOIN artifacts a ON a.id=aa.artifact_id
		WHERE aa.attempt_id=$1 AND aa.role=$2
	`, attemptID, "stage:"+stageID).Scan(
		&stageArtifactID,
		&stageSemantic,
		&stageStorage,
		&stageContent,
		&stageSize,
		&implementationSHA,
		&policySHA,
	); err != nil {
		return uuid.Nil, err
	}
	if err := a.Store.Verify(ctx, artifactstore.Object{
		ContentSHA256: stageContent,
		StorageKey:    stageStorage,
		SizeBytes:     stageSize,
	}); err != nil {
		return uuid.Nil, err
	}
	raw, err := a.Store.GetBytes(ctx, stageStorage)
	if err != nil {
		return uuid.Nil, err
	}
	var manifest stageResultManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return uuid.Nil, err
	}
	var selected *orchestration.EngineOutput
	for i := range manifest.Outputs {
		if manifest.Outputs[i].AuthorityClass == authorityClass {
			value := manifest.Outputs[i]
			selected = &value
			break
		}
	}
	if selected == nil {
		return uuid.Nil, fmt.Errorf("STAGE_OUTPUT_AUTHORITY_MISSING:%s:%s", stageID, authorityClass)
	}
	if err := a.Store.Verify(ctx, artifactstore.Object{
		ContentSHA256: selected.ContentSHA256,
		StorageKey:    selected.StorageKey,
		SizeBytes:     selected.SizeBytes,
	}); err != nil {
		return uuid.Nil, err
	}
	semanticSHA, err := semantic.JSONSHA256(map[string]any{
		"schema":                       "RealSaS.PromotedStageOutput.v1",
		"source_stage_semantic_sha256": stageSemantic,
		"source_authority_class":       authorityClass,
		"content_sha256":               selected.ContentSHA256,
		"product_role":                 productRole,
	})
	if err != nil {
		return uuid.Nil, err
	}

	var out uuid.UUID
	err = persistence.WithSerializableRetry(ctx, a.Pool, 5, func(tx pgx.Tx) error {
		typeID := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:artifact-type:"+artifactType+":"+schemaVersion))
		if _, err := tx.Exec(ctx, `
			INSERT INTO artifact_types(id,name,schema_version,domain)
			VALUES ($1,$2,$3,$4)
			ON CONFLICT (name,schema_version) DO NOTHING
		`, typeID, artifactType, schemaVersion, domain); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, "SELECT id FROM artifact_types WHERE name=$1 AND schema_version=$2", artifactType, schemaVersion).Scan(&typeID); err != nil {
			return err
		}
		var existingContent, existingStorage string
		var existingSize int64
		err := tx.QueryRow(ctx, `
			SELECT id,content_sha256,storage_key,size_bytes
			FROM artifacts
			WHERE artifact_type_id=$1 AND semantic_sha256=$2
			FOR SHARE
		`, typeID, semanticSHA).Scan(&out, &existingContent, &existingStorage, &existingSize)
		switch {
		case err == nil:
			if existingContent != selected.ContentSHA256 || existingStorage != selected.StorageKey || existingSize != selected.SizeBytes {
				return fmt.Errorf("PROMOTED_OUTPUT_NONDETERMINISM:%s", productRole)
			}
		case errors.Is(err, pgx.ErrNoRows):
			out = uuid.New()
			params, _ := json.Marshal(map[string]any{
				"source_stage_id":              stageID,
				"source_stage_semantic_sha256": stageSemantic,
				"source_authority_class":       authorityClass,
			})
			if _, err := tx.Exec(ctx, `
				INSERT INTO artifacts
				  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				   producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now())
			`, out, typeID, semanticSHA, selected.ContentSHA256, selected.StorageKey, selected.SizeBytes,
				stageID, implementationSHA, policySHA, params); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
				INSERT INTO artifact_inputs(artifact_id,input_role,ordinal,input_artifact_id)
				VALUES ($1,'source_stage_result',0,$2)
			`, out, stageArtifactID); err != nil {
				return err
			}
		default:
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO qualifications(id,artifact_id,qualification_type,result)
			VALUES ($1,$2,$3,'PASS')
			ON CONFLICT DO NOTHING
		`, uuid.New(), out, qualification); err != nil {
			return err
		}
		tag, err := tx.Exec(ctx, `
			INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
			VALUES ($1,$2,$3,'produced')
			ON CONFLICT (attempt_id,role) DO NOTHING
		`, attemptID, productRole, out)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			var existing uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT artifact_id FROM attempt_artifacts WHERE attempt_id=$1 AND role=$2", attemptID, productRole).Scan(&existing); err != nil {
				return err
			}
			if existing != out {
				return fmt.Errorf("ATTEMPT_PRODUCT_OUTPUT_DRIFT:%s", productRole)
			}
		}
		return nil
	})
	return out, err
}

func (a Activities) ResolveRenderRequest(ctx context.Context, in orchestration.RenderWorkflowInput) (orchestration.RenderResolution, error) {
	if err := a.Validate(); err != nil {
		return orchestration.RenderResolution{}, err
	}
	if a.Store == nil {
		return orchestration.RenderResolution{}, errors.New("resolve render requires artifact store")
	}
	renderID, err := uuid.Parse(in.RenderRequestID)
	if err != nil {
		return orchestration.RenderResolution{}, err
	}
	revisionID, err := uuid.Parse(in.ProductRevisionID)
	if err != nil {
		return orchestration.RenderResolution{}, err
	}
	subjectID, err := uuid.Parse(in.SubjectID)
	if err != nil {
		return orchestration.RenderResolution{}, err
	}
	var cached uuid.UUID
	err = a.Pool.QueryRow(ctx, `
		SELECT ro.artifact_id
		FROM render_outputs ro
		JOIN artifacts a ON a.id=ro.artifact_id
		JOIN qualifications q ON q.artifact_id=a.id
		WHERE ro.render_request_id=$1
		  AND q.qualification_type='RENDER_OUTPUT_VERIFIED'
		  AND q.result='PASS'
		ORDER BY ro.created_at DESC
		LIMIT 1
	`, renderID).Scan(&cached)
	if err == nil {
		value := cached.String()
		return orchestration.RenderResolution{CacheHit: true, OutputArtifactID: &value}, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return orchestration.RenderResolution{}, err
	}

	var dbSubject, dbRevision, motionID uuid.UUID
	var semanticSHA string
	var viewRaw, settingsRaw []byte
	if err := a.Pool.QueryRow(ctx, `
		SELECT subject_id,product_revision_id,motion_artifact_id,semantic_sha256,view_spec,render_settings
		FROM render_requests
		WHERE id=$1
	`, renderID).Scan(&dbSubject, &dbRevision, &motionID, &semanticSHA, &viewRaw, &settingsRaw); err != nil {
		return orchestration.RenderResolution{}, err
	}
	if dbSubject != subjectID || dbRevision != revisionID || semanticSHA != in.RenderRequestSemanticSHA256 {
		return orchestration.RenderResolution{}, errors.New("RENDER_REQUEST_IDENTITY_DRIFT")
	}
	var boundMotion uuid.UUID
	if err := a.Pool.QueryRow(ctx, `
		SELECT artifact_id
		FROM product_revision_artifacts
		WHERE product_revision_id=$1 AND role=$2
	`, revisionID, productMotionRole).Scan(&boundMotion); err != nil {
		return orchestration.RenderResolution{}, err
	}
	if boundMotion != motionID {
		return orchestration.RenderResolution{}, errors.New("RENDER_MOTION_NOT_BOUND_TO_PRODUCT_REVISION")
	}
	type objectRef struct {
		ID             uuid.UUID
		StorageKey     string
		ContentSHA256  string
		SizeBytes      int64
		SemanticSHA256 string
	}
	loadRole := func(role string) (objectRef, error) {
		var out objectRef
		err := a.Pool.QueryRow(ctx, `
			SELECT a.id,a.storage_key,a.content_sha256,a.size_bytes,a.semantic_sha256
			FROM product_revision_artifacts pra
			JOIN artifacts a ON a.id=pra.artifact_id
			WHERE pra.product_revision_id=$1 AND pra.role=$2
		`, revisionID, role).Scan(&out.ID, &out.StorageKey, &out.ContentSHA256, &out.SizeBytes, &out.SemanticSHA256)
		if err != nil {
			return objectRef{}, err
		}
		if err := a.Store.Verify(ctx, artifactstore.Object{
			ContentSHA256: out.ContentSHA256,
			StorageKey:    out.StorageKey,
			SizeBytes:     out.SizeBytes,
		}); err != nil {
			return objectRef{}, err
		}
		return out, nil
	}
	runtimeObject, err := loadRole(productRuntimeRole)
	if err != nil {
		return orchestration.RenderResolution{}, err
	}
	motionObject, err := loadRole(productMotionRole)
	if err != nil {
		return orchestration.RenderResolution{}, err
	}
	var viewSpec, settings map[string]any
	if err := json.Unmarshal(viewRaw, &viewSpec); err != nil {
		return orchestration.RenderResolution{}, err
	}
	if err := json.Unmarshal(settingsRaw, &settings); err != nil {
		return orchestration.RenderResolution{}, err
	}
	return orchestration.RenderResolution{
		RuntimeRequest: map[string]any{
			"schema":              "RealSaS.RenderRuntimeTailRequest.v1",
			"render_request_id":   renderID.String(),
			"product_revision_id": revisionID.String(),
			"subject_id":          subjectID.String(),
			"runtime_package": map[string]any{
				"artifact_id":     runtimeObject.ID.String(),
				"storage_key":     runtimeObject.StorageKey,
				"content_sha256":  runtimeObject.ContentSHA256,
				"size_bytes":      runtimeObject.SizeBytes,
				"semantic_sha256": runtimeObject.SemanticSHA256,
			},
			"motion": map[string]any{
				"artifact_id":     motionObject.ID.String(),
				"storage_key":     motionObject.StorageKey,
				"content_sha256":  motionObject.ContentSHA256,
				"size_bytes":      motionObject.SizeBytes,
				"semantic_sha256": motionObject.SemanticSHA256,
			},
			"view_spec":       viewSpec,
			"render_settings": settings,
		},
	}, nil
}

func (a Activities) CommitRenderOutput(ctx context.Context, req orchestration.RenderCommitRequest) (orchestration.RenderWorkflowResult, error) {
	if err := a.Validate(); err != nil {
		return orchestration.RenderWorkflowResult{}, err
	}
	if a.Store == nil {
		return orchestration.RenderWorkflowResult{}, errors.New("commit render requires artifact store")
	}
	if req.EngineResult.Status != "PASS" {
		return orchestration.RenderWorkflowResult{}, errors.New("CANNOT_COMMIT_FAILED_RENDER")
	}
	renderID, err := uuid.Parse(req.RenderRequestID)
	if err != nil {
		return orchestration.RenderWorkflowResult{}, err
	}
	storageKey := strings.TrimSpace(req.EngineResult.StorageKey)
	if storageKey == "" {
		storageKey = strings.TrimSpace(req.EngineResult.OutputPath)
	}
	if storageKey == "" {
		return orchestration.RenderWorkflowResult{}, errors.New("RENDER_OUTPUT_STORAGE_KEY_REQUIRED")
	}
	object := artifactstore.Object{
		ContentSHA256: req.EngineResult.ContentSHA256,
		StorageKey:    storageKey,
		SizeBytes:     req.EngineResult.SizeBytes,
	}
	if err := a.Store.Verify(ctx, object); err != nil {
		return orchestration.RenderWorkflowResult{}, err
	}

	var outputID uuid.UUID
	err = persistence.WithSerializableRetry(ctx, a.Pool, 5, func(tx pgx.Tx) error {
		var semanticSHA string
		var revisionID uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT semantic_sha256,product_revision_id FROM render_requests WHERE id=$1 FOR SHARE", renderID).Scan(&semanticSHA, &revisionID); err != nil {
			return err
		}
		typeID := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:artifact-type:"+renderArtifactType+":"+platformSchemaV1))
		if _, err := tx.Exec(ctx, `
			INSERT INTO artifact_types(id,name,schema_version,domain)
			VALUES ($1,$2,$3,'render')
			ON CONFLICT (name,schema_version) DO NOTHING
		`, typeID, renderArtifactType, platformSchemaV1); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, "SELECT id FROM artifact_types WHERE name=$1 AND schema_version=$2", renderArtifactType, platformSchemaV1).Scan(&typeID); err != nil {
			return err
		}
		var implementationSHA, policySHA string
		if err := tx.QueryRow(ctx, `
			SELECT ers.implementation_sha256,ers.policy_sha256
			FROM product_revisions pr
			JOIN attempts a ON a.id=pr.created_from_attempt_id
			JOIN engine_release_stages ers ON ers.release_id=a.engine_release_id
			WHERE pr.id=$1 AND ers.stage_id='44_NATIVE_PACKAGE_OPEN_PLAYBACK'
		`, revisionID).Scan(&implementationSHA, &policySHA); err != nil {
			return err
		}
		var existingContent, existingStorage string
		var existingSize int64
		err := tx.QueryRow(ctx, `
			SELECT id,content_sha256,storage_key,size_bytes
			FROM artifacts
			WHERE artifact_type_id=$1 AND semantic_sha256=$2
			FOR SHARE
		`, typeID, semanticSHA).Scan(&outputID, &existingContent, &existingStorage, &existingSize)
		switch {
		case err == nil:
			if existingContent != object.ContentSHA256 || existingStorage != object.StorageKey || existingSize != object.SizeBytes {
				return errors.New("RENDER_OUTPUT_NONDETERMINISM")
			}
		case errors.Is(err, pgx.ErrNoRows):
			outputID = uuid.New()
			params, _ := json.Marshal(map[string]any{"render_request_id": renderID.String()})
			if _, err := tx.Exec(ctx, `
				INSERT INTO artifacts
				  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				   producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now())
			`, outputID, typeID, semanticSHA, object.ContentSHA256, object.StorageKey, object.SizeBytes,
				orchestration.EngineRenderTailActivityName, implementationSHA, policySHA, params); err != nil {
				return err
			}
		default:
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO qualifications(id,artifact_id,qualification_type,result)
			VALUES ($1,$2,'RENDER_OUTPUT_VERIFIED','PASS')
			ON CONFLICT DO NOTHING
		`, uuid.New(), outputID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO render_outputs(render_request_id,artifact_id)
			VALUES ($1,$2)
			ON CONFLICT DO NOTHING
		`, renderID, outputID); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return orchestration.RenderWorkflowResult{}, err
	}
	return orchestration.RenderWorkflowResult{
		Status:           "PASS",
		CacheHit:         false,
		OutputArtifactID: outputID.String(),
	}, nil
}

func sortedBindings(rows []revisionBinding) []revisionBinding {
	out := append([]revisionBinding(nil), rows...)
	sort.Slice(out, func(i, j int) bool { return out[i].Role < out[j].Role })
	return out
}
