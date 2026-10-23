package control

import (
	"context"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

// capabilityModelRefsTx reads only the immutable ModelRelease snapshot already
// bound to the Attempt. Capability execution must never resolve a live model
// channel after the Attempt has been accepted.
func capabilityModelRefsTx(
	ctx context.Context,
	tx pgx.Tx,
	attemptID uuid.UUID,
) ([]orchestration.ModelReleaseRef, error) {
	selected, err := modelselection.LoadAttemptTx(ctx, tx, attemptID)
	if err != nil {
		return nil, err
	}
	return modelselection.EngineRefs(selected), nil
}
