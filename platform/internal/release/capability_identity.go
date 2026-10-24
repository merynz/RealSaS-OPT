package release

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// VerifyCapabilityImplementationBinding reconstructs the exact V2
// EngineRelease manifest identity from sealed Platform state and the product
// implementation map presented for capability-snapshot sealing. If the map is
// not the one that was bound into release_sha256, snapshot sealing fails closed.
func VerifyCapabilityImplementationBinding(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	implementations map[string]string,
) error {
	if pool == nil || releaseID == uuid.Nil {
		return fmt.Errorf("pool and release_id are required")
	}
	info, err := LoadInfo(ctx, pool, releaseID)
	if err != nil {
		return err
	}
	graph, pipelinePlanSHA, err := LoadGraph(ctx, pool, releaseID)
	if err != nil {
		return err
	}
	purpose, modelRequirements, err := LoadModelRequirements(ctx, pool, releaseID)
	if err != nil {
		return err
	}
	if purpose != info.Purpose {
		return fmt.Errorf("ENGINE_RELEASE_PURPOSE_DRIFT")
	}

	rows, err := pool.Query(ctx, `
		SELECT ordinal,stage_id,implementation_sha256,policy_sha256,semantic_parameters
		FROM engine_release_stages
		WHERE release_id=$1
		ORDER BY ordinal
	`, releaseID)
	if err != nil {
		return err
	}
	defer rows.Close()
	stages := []StageBinding{}
	for rows.Next() {
		var row StageBinding
		var raw []byte
		if err := rows.Scan(
			&row.Ordinal,
			&row.StageID,
			&row.ImplementationSHA256,
			&row.PolicySHA256,
			&raw,
		); err != nil {
			return err
		}
		if err := json.Unmarshal(raw, &row.SemanticParameters); err != nil {
			return err
		}
		stages = append(stages, row)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if len(stages) != graph.StageCount() {
		return ErrGraphDrift
	}
	copied := make(map[string]string, len(implementations))
	for id, digest := range implementations {
		copied[id] = digest
	}
	snapshot := graph.Snapshot()
	manifest := Manifest{
		ContractVersion:           "RealSaS.EngineReleaseManifest.v2",
		Name:                      info.Name,
		Purpose:                   info.Purpose,
		Stages:                    stages,
		DAG:                       &snapshot,
		PipelinePlanSHA256:        pipelinePlanSHA,
		ModelRequirements:         modelRequirements,
		CapabilityImplementations: copied,
	}
	recomputed, err := manifest.SHA256(graph)
	if err != nil {
		return err
	}
	if recomputed != info.ReleaseSHA256 {
		return fmt.Errorf(
			"ENGINE_RELEASE_CAPABILITY_IMPLEMENTATION_BINDING_DRIFT:%s!=%s",
			recomputed,
			info.ReleaseSHA256,
		)
	}
	return nil
}
