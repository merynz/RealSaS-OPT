package release

import "testing"

func TestInterventionRejectsReleaseScopeDrift(t *testing.T) {
	impact := Impact{DirectChanges: []DirectChange{{StageID: "RGB"}}, InvalidatedStageIDs: []string{"RGB", "RENDER"}, UnchangedStageIDs: []string{"M", "G", "W"}}
	for _, c := range []Intervention{
		{DirectChangedStageIDs: []string{"RGB"}, PreservedStageIDs: []string{"M", "G", "W"}},
		{DirectChangedStageIDs: []string{"RGB"}},
	} {
		if err := c.Validate(impact); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []Intervention{
		{DirectChangedStageIDs: []string{"OTHER"}},
		{DirectChangedStageIDs: []string{"RGB", "RGB"}},
		{DirectChangedStageIDs: []string{"RGB", "M"}},
		{DirectChangedStageIDs: []string{"RGB"}, PreservedStageIDs: []string{"RENDER"}},
		{DirectChangedStageIDs: []string{"RGB"}, PreservedStageIDs: []string{"ABSENT"}},
		{DirectChangedStageIDs: []string{"RGB"}, PreservedStageIDs: []string{"M", "M"}},
	} {
		if err := c.Validate(impact); err == nil {
			t.Fatalf("accepted scope drift: %+v", c)
		}
	}
}
