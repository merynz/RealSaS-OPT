package attempt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

// FrozenIntervention is part of the immutable attempt spec and impact event.
// Artifact IDs are captured when the attempt starts, not looked up from a moving parent later.
type FrozenIntervention struct {
	Contract            release.Intervention `json:"contract"`
	ParentAttemptID     uuid.UUID            `json:"parent_attempt_id"`
	FrozenStageIDs      []string             `json:"frozen_stage_ids"`
	BaselineArtifactIDs map[string]uuid.UUID `json:"baseline_artifact_ids"`
}

func LoadIntervention(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) (*FrozenIntervention, error) {
	var payload []byte
	err := pool.QueryRow(ctx, `SELECT payload FROM attempt_events
		WHERE attempt_id=$1 AND event_type='CODE_CHANGE_IMPACT_RESOLVED'
		ORDER BY id LIMIT 1`, attemptID).Scan(&payload)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil // Product and legacy research attempts have no intervention contract.
	}
	if err != nil {
		return nil, err
	}
	var event struct {
		Intervention *FrozenIntervention `json:"intervention"`
	}
	if err := json.Unmarshal(payload, &event); err != nil {
		return nil, fmt.Errorf("INTERVENTION_EVENT_INVALID: %w", err)
	}
	return event.Intervention, nil
}

func (c *FrozenIntervention) CheckExecution(stageID string) error {
	if c != nil && slices.Contains(c.FrozenStageIDs, stageID) {
		return fmt.Errorf("INTERVENTION_FROZEN_STAGE_EXECUTION: %s", stageID)
	}
	return nil
}

func (c *FrozenIntervention) CheckReuse(stageID string, artifactID uuid.UUID) error {
	if c != nil && slices.Contains(c.FrozenStageIDs, stageID) {
		if id, ok := c.BaselineArtifactIDs[stageID]; !ok || id != artifactID {
			return fmt.Errorf("INTERVENTION_EXACT_REUSE_REQUIRED: %s", stageID)
		}
	}
	return nil
}
