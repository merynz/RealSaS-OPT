package release

import (
	"encoding/json"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
	"reflect"
	"testing"
)

func TestModelHeadAndEvaluationChangesHaveNoSacredFrozenModules(t *testing.T) {
	stages := []stagegraph.Stage{
		{Ordinal: 1, ID: "MODEL_TRUNK"}, {Ordinal: 2, ID: "GEOMETRY_HEAD", DependsOn: []string{"MODEL_TRUNK"}},
		{Ordinal: 3, ID: "APPEARANCE_HEAD", DependsOn: []string{"MODEL_TRUNK"}},
		{Ordinal: 4, ID: "TREE_MODEL"}, {Ordinal: 5, ID: "SKIN_MODEL", DependsOn: []string{"GEOMETRY_HEAD", "TREE_MODEL"}},
		{Ordinal: 6, ID: "UNSEEN_EVALUATION", DependsOn: []string{"GEOMETRY_HEAD", "SKIN_MODEL", "APPEARANCE_HEAD"}},
	}
	raw, _ := json.Marshal(stagegraph.Snapshot{StageCount: len(stages), Stages: stages})
	g, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		root string
		want []string
	}{
		{"APPEARANCE_HEAD", []string{"APPEARANCE_HEAD", "UNSEEN_EVALUATION"}},
		{"GEOMETRY_HEAD", []string{"GEOMETRY_HEAD", "SKIN_MODEL", "UNSEEN_EVALUATION"}},
		{"TREE_MODEL", []string{"TREE_MODEL", "SKIN_MODEL", "UNSEEN_EVALUATION"}},
		{"SKIN_MODEL", []string{"SKIN_MODEL", "UNSEEN_EVALUATION"}},
		{"UNSEEN_EVALUATION", []string{"UNSEEN_EVALUATION"}},
		{"MODEL_TRUNK", []string{"MODEL_TRUNK", "GEOMETRY_HEAD", "APPEARANCE_HEAD", "SKIN_MODEL", "UNSEEN_EVALUATION"}},
	} {
		t.Run(tc.root, func(t *testing.T) {
			old, next := sameVersions(g), sameVersions(g)
			v := next[tc.root]
			v.ImplementationSHA256 = "replacement"
			next[tc.root] = v
			impact, err := Compare(g, old, next)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(impact.InvalidatedStageIDs, tc.want) {
				t.Fatalf("impact=%+v", impact)
			}
		})
	}
}
