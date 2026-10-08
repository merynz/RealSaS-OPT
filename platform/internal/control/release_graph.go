package control

import (
	"context"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func (a Activities) attemptGraph(ctx context.Context, attemptID uuid.UUID) (*stagegraph.Graph, error) {
	var releaseID uuid.UUID
	if err := a.Pool.QueryRow(ctx, "SELECT engine_release_id FROM attempts WHERE id=$1", attemptID).Scan(&releaseID); err != nil {
		return nil, err
	}
	g, _, err := release.LoadGraph(ctx, a.Pool, releaseID)
	return g, err
}
