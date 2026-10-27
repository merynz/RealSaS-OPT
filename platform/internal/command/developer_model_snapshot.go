package command

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
)

// snapshotDeveloperTargetModels freezes only the model channels consumed by the
// resolved DEVELOPER_RUN capability closure. The EngineRelease still owns the
// superset contract; this function verifies every target requirement against
// that sealed release contract before selecting the live channel.
//
// compile_candidate attempts intentionally do not use this path: their
// release-wide model snapshot remains enforced by the database trigger.
func snapshotDeveloperTargetModels(
	ctx context.Context,
	tx pgx.Tx,
	attemptID uuid.UUID,
	releaseID uuid.UUID,
	requirements []capability.ProductModelRequirement,
) error {
	for _, requirement := range requirements {
		var releaseArchitecture string
		var channel string
		if err := tx.QueryRow(ctx, `
			SELECT architecture_id,channel
			FROM engine_release_model_requirements
			WHERE release_id=$1 AND family=$2
			FOR SHARE
		`, releaseID, requirement.Family).Scan(&releaseArchitecture, &channel); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return fmt.Errorf("DEVELOPER_TARGET_MODEL_REQUIREMENT_MISSING:%s", requirement.Family)
			}
			return err
		}
		if releaseArchitecture != requirement.ArchitectureID {
			return fmt.Errorf(
				"DEVELOPER_TARGET_MODEL_ARCHITECTURE_DRIFT:%s:%s!=%s",
				requirement.Family, releaseArchitecture, requirement.ArchitectureID,
			)
		}

		var modelReleaseID uuid.UUID
		var selectedArchitecture string
		var lifecycleState string
		var lockVersion int64
		if err := tx.QueryRow(ctx, `
			SELECT mc.model_release_id,mr.architecture_id,mr.lifecycle_state,mc.lock_version
			FROM model_channels mc
			JOIN model_releases mr ON mr.id=mc.model_release_id
			WHERE mc.family=$1 AND mc.channel=$2
			FOR SHARE OF mc,mr
		`, requirement.Family, channel).Scan(
			&modelReleaseID, &selectedArchitecture, &lifecycleState, &lockVersion,
		); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return fmt.Errorf("MODEL_SELECTION_CHANNEL_MISSING:%s:%s", requirement.Family, channel)
			}
			return err
		}
		if selectedArchitecture != requirement.ArchitectureID {
			return fmt.Errorf(
				"MODEL_SELECTION_ARCHITECTURE_MISMATCH:%s:%s!=%s",
				requirement.Family, selectedArchitecture, requirement.ArchitectureID,
			)
		}
		if channel == "PRODUCTION" && lifecycleState != "PRODUCTION" {
			return fmt.Errorf("MODEL_SELECTION_NOT_PRODUCTION:%s:%s", requirement.Family, lifecycleState)
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_model_releases(
				attempt_id,family,model_release_id,selected_channel,selected_channel_lock_version
			) VALUES ($1,$2,$3,$4,$5)
		`, attemptID, requirement.Family, modelReleaseID, channel, lockVersion); err != nil {
			return err
		}
	}
	return nil
}
