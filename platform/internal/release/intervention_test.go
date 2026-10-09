package release

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestInputInterventionInvalidatesOnlyDeclaredConsumers(t *testing.T) {
	raw, _ := json.Marshal(stagegraph.Snapshot{StageCount: 3, Stages: []stagegraph.Stage{
		{Ordinal: 1, ID: "M", ManifestKeys: []string{"geometry"}},
		{Ordinal: 2, ID: "RGB", ManifestKeys: []string{"appearance"}},
		{Ordinal: 3, ID: "RENDER", DependsOn: []string{"M", "RGB"}},
	}})
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	c := Intervention{ChangedInputRoles: []string{"manifest:appearance"}, PreservedStageIDs: []string{"M"}}
	impact, consumers, err := c.InputImpact(g, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(consumers, []string{"RGB"}) || !reflect.DeepEqual(impact.InvalidatedStageIDs, []string{"RGB", "RENDER"}) {
		t.Fatalf("input impact=%+v consumers=%v", impact, consumers)
	}
	if err := c.Validate(impact); err != nil {
		t.Fatal(err)
	}
	c.ChangedInputRoles = []string{"source"}
	impact, _, err = c.InputImpact(g, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Validate(impact); err == nil {
		t.Fatal("unsegmented source change preserved mechanics")
	}
	c.ChangedInputRoles = []string{"manifest:appearance", "manifest:appearance"}
	if _, _, err := c.InputImpact(g, Impact{}); err == nil {
		t.Fatal("duplicate roles accepted")
	}
}

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
