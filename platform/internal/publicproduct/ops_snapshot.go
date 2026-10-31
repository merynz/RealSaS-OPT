package publicproduct

import (
	"context"
	"sort"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type ProductOpsStateCount struct {
	State string `json:"state"`
	Count int64  `json:"count"`
}

type ProductOpsStudioState struct {
	ActiveProjects                int64 `json:"active_projects"`
	AuthoringRevisions            int64 `json:"authoring_revisions"`
	HeadsRequiringRequalification int64 `json:"heads_requiring_requalification"`
}

type ProductOpsSnapshot struct {
	SchemaVersion             string                   `json:"schema_version"`
	GeneratedAt               time.Time                `json:"generated_at"`
	Ready                     bool                     `json:"ready"`
	SourceCodeSHA             string                   `json:"source_code_sha"`
	Preflight                 DeploymentPreflightState `json:"preflight"`
	CompileSessions           []ProductOpsStateCount   `json:"compile_sessions"`
	WorkspaceMaterializations []ProductOpsStateCount   `json:"workspace_materializations"`
	Studio                    ProductOpsStudioState    `json:"studio"`
}

// ProductOpsSnapshotState is the production-facing operator read model. It is
// intentionally product-only: research experiments/campaigns and internal
// revision/artifact identities are excluded. The snapshot is observational;
// it never repairs, promotes, leases, or mutates product state.
func ProductOpsSnapshotState(ctx context.Context, pool *pgxpool.Pool, codeSHA string) (ProductOpsSnapshot, error) {
	preflight, err := ProductDeploymentPreflightState(ctx, pool, codeSHA)
	if err != nil {
		return ProductOpsSnapshot{}, err
	}
	compileSessions, err := readProductStateCounts(ctx, pool, `
		SELECT state, count(*)
		FROM public_product_compile_sessions
		GROUP BY state
	`)
	if err != nil {
		return ProductOpsSnapshot{}, err
	}
	workspaceMaterializations, err := readProductStateCounts(ctx, pool, `
		SELECT state, count(*)
		FROM public_product_workspace_materializations
		GROUP BY state
	`)
	if err != nil {
		return ProductOpsSnapshot{}, err
	}

	var studio ProductOpsStudioState
	if err := pool.QueryRow(ctx, `
		SELECT
			(SELECT count(*) FROM studio_projects WHERE state='ACTIVE'),
			(SELECT count(*) FROM studio_authoring_revisions),
			(SELECT count(*)
			 FROM studio_project_heads h
			 JOIN studio_authoring_revisions ar ON ar.id=h.authoring_revision_id
			 WHERE ar.requires_requalification=true)
	`).Scan(
		&studio.ActiveProjects,
		&studio.AuthoringRevisions,
		&studio.HeadsRequiringRequalification,
	); err != nil {
		return ProductOpsSnapshot{}, err
	}

	return ProductOpsSnapshot{
		SchemaVersion:             "RealSaS.ProductOpsSnapshot.v1",
		GeneratedAt:               time.Now().UTC(),
		Ready:                     preflight.Ready,
		SourceCodeSHA:             codeSHA,
		Preflight:                 preflight,
		CompileSessions:           compileSessions,
		WorkspaceMaterializations: workspaceMaterializations,
		Studio:                    studio,
	}, nil
}

func readProductStateCounts(ctx context.Context, pool *pgxpool.Pool, query string) ([]ProductOpsStateCount, error) {
	rows, err := pool.Query(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []ProductOpsStateCount{}
	for rows.Next() {
		var item ProductOpsStateCount
		if err := rows.Scan(&item.State, &item.Count); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].State < out[j].State })
	return out, nil
}
