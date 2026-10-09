package resolver

import (
	"fmt"
	"slices"

	"github.com/google/uuid"
)

type PreservedArtifact struct {
	StageID        string    `json:"stage_id"`
	ArtifactID     uuid.UUID `json:"artifact_id"`
	SemanticSHA256 string    `json:"semantic_sha256"`
}

type InterventionReceipt struct {
	PreservedArtifacts    []PreservedArtifact `json:"preserved_artifacts"`
	OutsideTargetStageIDs []string            `json:"outside_target_stage_ids"`
	ExecuteStageIDs       []string            `json:"execute_stage_ids"`
}

// VerifyIntervention fails before any engine work if a frozen stage would run,
// or would reuse an artifact other than the one captured from the baseline.
func VerifyIntervention(plan Plan, frozen []string, baseline map[string]uuid.UUID) (InterventionReceipt, error) {
	receipt := InterventionReceipt{ExecuteStageIDs: plan.ExecuteStageIDs()}
	inside := map[string]bool{}
	for _, row := range plan.Stages {
		inside[row.StageID] = true
		if !slices.Contains(frozen, row.StageID) {
			continue
		}
		id, ok := baseline[row.StageID]
		if !ok || id == uuid.Nil {
			return InterventionReceipt{}, fmt.Errorf("INTERVENTION_BASELINE_ARTIFACT_MISSING: %s", row.StageID)
		}
		if row.Action != Reuse || row.ReusableArtifactID == nil || *row.ReusableArtifactID != id {
			return InterventionReceipt{}, fmt.Errorf("INTERVENTION_EXACT_REUSE_REQUIRED: %s", row.StageID)
		}
		receipt.PreservedArtifacts = append(receipt.PreservedArtifacts, PreservedArtifact{
			StageID: row.StageID, ArtifactID: id, SemanticSHA256: row.ExpectedSemanticSHA256,
		})
	}
	for _, id := range frozen {
		if !inside[id] {
			receipt.OutsideTargetStageIDs = append(receipt.OutsideTargetStageIDs, id)
		}
	}
	return receipt, nil
}
