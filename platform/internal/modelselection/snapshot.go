package modelselection

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

// SnapshotAttemptRequirementsTx binds exact ModelReleases to one Attempt from
// the channels sealed into its EngineRelease contract. The caller supplies only
// the target-scoped release requirements that are behaviorally necessary for
// the resolved capability closure. Selection occurs in the caller's serializable
// transaction; later execution reads only attempt_model_releases and never a
// floating channel.
func SnapshotAttemptRequirementsTx(
	ctx context.Context,
	tx pgx.Tx,
	attemptID uuid.UUID,
	requirements []release.ModelRequirement,
) error {
	if tx == nil || attemptID == uuid.Nil {
		return errors.New("MODEL_SELECTION_SNAPSHOT_ATTEMPT_AND_TX_REQUIRED")
	}
	ordered := append([]release.ModelRequirement(nil), requirements...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Family < ordered[j].Family })
	seen := map[string]struct{}{}
	for _, requirement := range ordered {
		if requirement.Family == "" || requirement.ArchitectureID == "" {
			return errors.New("MODEL_SELECTION_REQUIREMENT_IDENTITY_REQUIRED")
		}
		if requirement.Channel != "RESEARCH" && requirement.Channel != "CANDIDATE" && requirement.Channel != "PRODUCTION" {
			return fmt.Errorf("MODEL_SELECTION_CHANNEL_INVALID:%s:%s", requirement.Family, requirement.Channel)
		}
		if _, duplicate := seen[requirement.Family]; duplicate {
			return fmt.Errorf("MODEL_SELECTION_REQUIREMENT_DUPLICATE:%s", requirement.Family)
		}
		seen[requirement.Family] = struct{}{}

		var modelReleaseID uuid.UUID
		var lockVersion int64
		var architectureID, lifecycleState string
		err := tx.QueryRow(ctx, `
			SELECT mc.model_release_id,mc.lock_version,mr.architecture_id,mr.lifecycle_state
			FROM model_channels mc
			JOIN model_releases mr ON mr.id=mc.model_release_id
			WHERE mc.family=$1 AND mc.channel=$2
			FOR SHARE OF mc,mr
		`, requirement.Family, requirement.Channel).Scan(
			&modelReleaseID, &lockVersion, &architectureID, &lifecycleState,
		)
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("MODEL_SELECTION_CHANNEL_MISSING:%s:%s", requirement.Family, requirement.Channel)
		}
		if err != nil {
			return err
		}
		if architectureID != requirement.ArchitectureID {
			return fmt.Errorf("MODEL_SELECTION_ARCHITECTURE_MISMATCH:%s:%s!=%s", requirement.Family, architectureID, requirement.ArchitectureID)
		}
		if requirement.Channel == "PRODUCTION" && lifecycleState != "PRODUCTION" {
			return fmt.Errorf("MODEL_SELECTION_NOT_PRODUCTION:%s:%s", requirement.Family, lifecycleState)
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_model_releases(
				attempt_id,family,model_release_id,
				selected_channel,selected_channel_lock_version
			) VALUES ($1,$2,$3,$4,$5)
		`, attemptID, requirement.Family, modelReleaseID, requirement.Channel, lockVersion); err != nil {
			return err
		}
	}
	return nil
}
