package control

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
)

// capabilityModelRefs selects the exact immutable ModelRelease subset owned by
// one product capability. The Attempt remains the global immutable selection
// authority; this function only prevents unrelated families from crossing a
// capability execution boundary.
func capabilityModelRefs(
	descriptor capability.Descriptor,
	selected []modelselection.Selection,
) ([]orchestration.ModelReleaseRef, error) {
	if !strings.HasPrefix(descriptor.ID, "product.") {
		return modelselection.EngineRefs(selected), nil
	}
	required, err := capability.DescriptorRequiredModels(descriptor)
	if err != nil {
		return nil, err
	}
	byFamily := make(map[string]modelselection.Selection, len(selected))
	for _, row := range selected {
		if _, exists := byFamily[row.Family]; exists {
			return nil, fmt.Errorf("CAPABILITY_MODEL_SELECTION_DUPLICATE_FAMILY:%s", row.Family)
		}
		byFamily[row.Family] = row
	}
	filtered := make([]modelselection.Selection, 0, len(required))
	for _, want := range required {
		got, ok := byFamily[want.Family]
		if !ok {
			return nil, fmt.Errorf("CAPABILITY_MODEL_RELEASE_REQUIRED:%s:%s", descriptor.ID, want.Family)
		}
		if got.ArchitectureID != want.ArchitectureID {
			return nil, fmt.Errorf(
				"CAPABILITY_MODEL_ARCHITECTURE_DRIFT:%s:%s:%s!=%s",
				descriptor.ID,
				want.Family,
				got.ArchitectureID,
				want.ArchitectureID,
			)
		}
		filtered = append(filtered, got)
	}
	return modelselection.EngineRefs(filtered), nil
}

// capabilityModelRefsTx reads only the immutable ModelRelease snapshot already
// bound to the Attempt. Capability execution must never resolve a live model
// channel after the Attempt has been accepted.
func capabilityModelRefsTx(
	ctx context.Context,
	tx pgx.Tx,
	attemptID uuid.UUID,
	descriptor capability.Descriptor,
) ([]orchestration.ModelReleaseRef, error) {
	selected, err := modelselection.LoadAttemptTx(ctx, tx, attemptID)
	if err != nil {
		return nil, err
	}
	return capabilityModelRefs(descriptor, selected)
}
