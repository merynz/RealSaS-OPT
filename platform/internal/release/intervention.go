package release

import (
	"fmt"
	"slices"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

// Intervention declares the intended release change before an attempt is opened.
// Every unchanged stage is frozen, even if it is not explicitly listed here.
type Intervention struct {
	DirectChangedStageIDs []string `json:"direct_changed_stage_ids"`
	PreservedStageIDs     []string `json:"preserved_stage_ids,omitempty"`
	ChangedInputRoles     []string `json:"changed_input_roles,omitempty"`
}

// InputImpact adds descendants of declared input consumers. DirectChanges
// continues to describe release identity changes, separately from inputs.
func (c Intervention) InputImpact(g *stagegraph.Graph, impact Impact) (Impact, []string, error) {
	seen := map[string]bool{}
	for _, role := range c.ChangedInputRoles {
		if strings.TrimSpace(role) != role || role == "" || strings.HasPrefix(role, "subject:") || seen[role] {
			return Impact{}, nil, fmt.Errorf("INTERVENTION_INPUT_ROLE_INVALID: %q", role)
		}
		seen[role] = true
	}
	consumers := []string{}
	for _, stage := range g.Stages() {
		for _, role := range c.ChangedInputRoles {
			if stage.ConsumesInput("subject:" + role) {
				consumers = append(consumers, stage.ID)
				break
			}
		}
	}
	inputAffected, err := g.DescendantsIncluding(consumers...)
	if err != nil {
		return Impact{}, nil, err
	}
	invalid := map[string]bool{}
	for _, id := range impact.InvalidatedStageIDs {
		invalid[id] = true
	}
	for _, id := range inputAffected {
		invalid[id] = true
	}
	impact.InvalidatedStageIDs = []string{}
	impact.UnchangedStageIDs = []string{}
	for _, stage := range g.Stages() {
		if invalid[stage.ID] {
			impact.InvalidatedStageIDs = append(impact.InvalidatedStageIDs, stage.ID)
		} else {
			impact.UnchangedStageIDs = append(impact.UnchangedStageIDs, stage.ID)
		}
	}
	return impact, consumers, nil
}

func (c Intervention) Validate(impact Impact) error {
	expected := make([]string, 0, len(impact.DirectChanges))
	for _, change := range impact.DirectChanges {
		expected = append(expected, change.StageID)
	}
	declared := slices.Clone(c.DirectChangedStageIDs)
	slices.Sort(expected)
	slices.Sort(declared)
	if !slices.Equal(expected, declared) {
		return fmt.Errorf("INTERVENTION_DIRECT_CHANGE_DRIFT: declared=%v actual=%v", declared, expected)
	}
	seen := map[string]bool{}
	for _, id := range c.PreservedStageIDs {
		if seen[id] || !slices.Contains(impact.UnchangedStageIDs, id) {
			return fmt.Errorf("INTERVENTION_PRESERVED_STAGE_DRIFT: %s", id)
		}
		seen[id] = true
	}
	return nil
}
