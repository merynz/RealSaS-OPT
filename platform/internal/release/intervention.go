package release

import (
	"fmt"
	"slices"
)

// Intervention declares the intended release change before an attempt is opened.
// Every unchanged stage is frozen, even if it is not explicitly listed here.
type Intervention struct {
	DirectChangedStageIDs []string `json:"direct_changed_stage_ids"`
	PreservedStageIDs     []string `json:"preserved_stage_ids,omitempty"`
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
