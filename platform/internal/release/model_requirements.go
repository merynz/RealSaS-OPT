package release

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// LoadModelRequirements returns the exact learned-model contract sealed into an
// EngineRelease. Capability terrain validation uses this read path rather than
// re-querying release internals in multiple packages.
func LoadModelRequirements(ctx context.Context, pool *pgxpool.Pool, releaseID uuid.UUID) (Purpose, []ModelRequirement, error) {
	if pool == nil || releaseID == uuid.Nil {
		return "", nil, fmt.Errorf("pool and release_id are required")
	}
	var purpose Purpose
	if err := pool.QueryRow(ctx, "SELECT purpose FROM engine_releases WHERE id=$1", releaseID).Scan(&purpose); err != nil {
		return "", nil, err
	}
	rows, err := pool.Query(ctx, `
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
