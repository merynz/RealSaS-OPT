package release

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func loadModelRequirementsRows(ctx context.Context, q interface {
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
}, releaseID uuid.UUID) (Purpose, []ModelRequirement, error) {
	if releaseID == uuid.Nil {
		return "", nil, fmt.Errorf("release_id is required")
	}
	var purpose Purpose
	if err := q.QueryRow(ctx, "SELECT purpose FROM engine_releases WHERE id=$1", releaseID).Scan(&purpose); err != nil {
		return "", nil, err
	}
	rows, err := q.Query(ctx, `
		SELECT family,architecture_id,channel
		FROM engine_release_model_requirements
		WHERE release_id=$1
		ORDER BY family
	`, releaseID)
	if err != nil {
		return "", nil, err
	}
	defer rows.Close()
	out := []ModelRequirement{}
	for rows.Next() {
		var row ModelRequirement
		if err := rows.Scan(&row.Family, &row.ArchitectureID, &row.Channel); err != nil {
			return "", nil, err
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return "", nil, err
	}
	return purpose, out, nil
}

// LoadModelRequirements returns the exact learned-model contract sealed into an
// EngineRelease. Capability terrain validation uses this read path rather than
// re-querying release internals in multiple packages.
func LoadModelRequirements(ctx context.Context, pool *pgxpool.Pool, releaseID uuid.UUID) (Purpose, []ModelRequirement, error) {
	if pool == nil {
		return "", nil, fmt.Errorf("pool is required")
	}
	return loadModelRequirementsRows(ctx, pool, releaseID)
}

// LoadModelRequirementsTx is the transactional counterpart used when an
// Attempt snapshots immutable model selections. The release contract and model
// channel selection can therefore be checked inside one serializable boundary.
func LoadModelRequirementsTx(ctx context.Context, tx pgx.Tx, releaseID uuid.UUID) (Purpose, []ModelRequirement, error) {
	if tx == nil {
		return "", nil, fmt.Errorf("transaction is required")
	}
	return loadModelRequirementsRows(ctx, tx, releaseID)
}
