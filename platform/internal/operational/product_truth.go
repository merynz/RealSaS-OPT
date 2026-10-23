package operational

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// loadCurrentProducts is the single read path for promoted product truth.
// Operational status, baseline resolution and future agent context surfaces use
// the same query so exact EngineRelease, ModelRelease and artifact lineage
// cannot drift between read APIs.
func loadCurrentProducts(ctx context.Context, pool *pgxpool.Pool, subjectID *uuid.UUID) ([]CurrentProduct, error) {
	query := `
		SELECT c.subject_id,s.slug,c.product_revision_id,r.revision_number,r.manifest_sha256,
		       r.engine_release_id,er.release_sha256,r.capability_set_sha256,c.updated_at
		FROM subject_current_revision c
		JOIN subjects s ON s.id=c.subject_id
		JOIN product_revisions r ON r.id=c.product_revision_id
		LEFT JOIN engine_releases er ON er.id=r.engine_release_id`
	args := []any{}
	if subjectID != nil {
		query += " WHERE c.subject_id=$1"
		args = append(args, *subjectID)
	}
	query += " ORDER BY s.slug,c.subject_id"

	rows, err := pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	var out []CurrentProduct
	productIndex := map[uuid.UUID]int{}
	for rows.Next() {
		var record CurrentProduct
		if err := rows.Scan(
			&record.SubjectID,
			&record.SubjectSlug,
			&record.ProductRevisionID,
			&record.RevisionNumber,
			&record.ManifestSHA256,
			&record.EngineReleaseID,
			&record.EngineReleaseSHA256,
			&record.CapabilitySetSHA256,
			&record.UpdatedAt,
		); err != nil {
			rows.Close()
			return nil, err
		}
		record.Models = []ProductModelBinding{}
		record.Artifacts = []ProductArtifactBinding{}
		productIndex[record.ProductRevisionID] = len(out)
		out = append(out, record)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return []CurrentProduct{}, nil
	}

	modelQuery := `
		SELECT prm.product_revision_id,prm.family,prm.model_release_id,
		       mr.architecture_id,mr.checkpoint_artifact_id,a.content_sha256,mr.lifecycle_state
		FROM product_revision_model_releases prm
		JOIN subject_current_revision c ON c.product_revision_id=prm.product_revision_id
		JOIN model_releases mr ON mr.id=prm.model_release_id
		JOIN artifacts a ON a.id=mr.checkpoint_artifact_id`
	modelArgs := []any{}
	if subjectID != nil {
		modelQuery += " WHERE c.subject_id=$1"
		modelArgs = append(modelArgs, *subjectID)
	}
	modelQuery += " ORDER BY prm.product_revision_id,prm.family"
	modelRows, err := pool.Query(ctx, modelQuery, modelArgs...)
	if err != nil {
		return nil, err
	}
	for modelRows.Next() {
		var revisionID uuid.UUID
		var binding ProductModelBinding
		if err := modelRows.Scan(
			&revisionID,
			&binding.Family,
			&binding.ModelReleaseID,
			&binding.ArchitectureID,
			&binding.CheckpointArtifactID,
			&binding.CheckpointContentSHA256,
			&binding.LifecycleState,
		); err != nil {
			modelRows.Close()
			return nil, err
		}
		index, ok := productIndex[revisionID]
		if !ok {
			modelRows.Close()
			return nil, fmt.Errorf("CURRENT_PRODUCT_MODEL_BINDING_WITHOUT_CURRENT_REVISION:%s", revisionID)
		}
		out[index].Models = append(out[index].Models, binding)
	}
	modelRows.Close()
	if err := modelRows.Err(); err != nil {
		return nil, err
	}

	artifactQuery := `
		SELECT pra.product_revision_id,pra.role,a.id,t.name,t.schema_version,
		       a.semantic_sha256,a.content_sha256
		FROM product_revision_artifacts pra
		JOIN subject_current_revision c ON c.product_revision_id=pra.product_revision_id
		JOIN artifacts a ON a.id=pra.artifact_id
		JOIN artifact_types t ON t.id=a.artifact_type_id`
	artifactArgs := []any{}
	if subjectID != nil {
		artifactQuery += " WHERE c.subject_id=$1"
		artifactArgs = append(artifactArgs, *subjectID)
	}
	artifactQuery += " ORDER BY pra.product_revision_id,pra.role"
	artifactRows, err := pool.Query(ctx, artifactQuery, artifactArgs...)
	if err != nil {
		return nil, err
	}
	for artifactRows.Next() {
		var revisionID uuid.UUID
		var binding ProductArtifactBinding
		if err := artifactRows.Scan(
			&revisionID,
			&binding.Role,
			&binding.ArtifactID,
			&binding.ArtifactType,
			&binding.SchemaVersion,
			&binding.SemanticSHA256,
			&binding.ContentSHA256,
		); err != nil {
			artifactRows.Close()
			return nil, err
		}
		index, ok := productIndex[revisionID]
		if !ok {
			artifactRows.Close()
			return nil, fmt.Errorf("CURRENT_PRODUCT_ARTIFACT_BINDING_WITHOUT_CURRENT_REVISION:%s", revisionID)
		}
		out[index].Artifacts = append(out[index].Artifacts, binding)
	}
	artifactRows.Close()
	if err := artifactRows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
