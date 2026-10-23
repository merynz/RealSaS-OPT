package attempt

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
)

func resolveResearchCapabilityImpact(
	ctx context.Context,
	pool *pgxpool.Pool,
	baselineReleaseID uuid.UUID,
	candidateReleaseID uuid.UUID,
) (*capability.CapabilityImpact, error) {
	baseline, baselineErr := capability.LoadReleaseSnapshot(ctx, pool, baselineReleaseID)
	candidate, candidateErr := capability.LoadReleaseSnapshot(ctx, pool, candidateReleaseID)

	baselineMissing := errors.Is(baselineErr, capability.ErrCapabilitySnapshotMissing)
	candidateMissing := errors.Is(candidateErr, capability.ErrCapabilitySnapshotMissing)
	if baselineMissing && candidateMissing {
		// Legacy releases remain comparable through the historical stage graph.
		return nil, nil
	}
	if baselineErr != nil && !baselineMissing {
		return nil, baselineErr
	}
	if candidateErr != nil && !candidateMissing {
		return nil, candidateErr
	}
	if baselineMissing != candidateMissing {
		return nil, fmt.Errorf(
			"RESEARCH_CAPABILITY_SNAPSHOT_PARITY_REQUIRED:baseline_missing=%t:candidate_missing=%t",
			baselineMissing,
			candidateMissing,
		)
	}
	impact, err := capability.CompareReleaseSnapshots(baseline, candidate)
	if err != nil {
		return nil, err
	}
	return &impact, nil
}
