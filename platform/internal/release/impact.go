package release

import (
	"fmt"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type StageVersion struct {
	ImplementationSHA256 string
	PolicySHA256         string
	ParametersSHA256     string
}

type DirectChange struct {
	StageID               string
	ImplementationChanged bool
	PolicyChanged         bool
	ParametersChanged     bool
}

type Impact struct {
	DirectChanges       []DirectChange
	InvalidatedStageIDs []string
	UnchangedStageIDs   []string
}

func Compare(g *stagegraph.Graph, baseline, candidate map[string]StageVersion) (Impact, error) {
	var direct []DirectChange
	var roots []string
	for _, s := range g.Stages() {
		a, okA := baseline[s.ID]
		b, okB := candidate[s.ID]
		if !okA || !okB {
			return Impact{}, fmt.Errorf("missing stage version %s", s.ID)
		}
		change := DirectChange{
			StageID:               s.ID,
			ImplementationChanged: a.ImplementationSHA256 != b.ImplementationSHA256,
			PolicyChanged:         a.PolicySHA256 != b.PolicySHA256,
			ParametersChanged:     a.ParametersSHA256 != b.ParametersSHA256,
		}
		if change.ImplementationChanged || change.PolicyChanged || change.ParametersChanged {
			direct = append(direct, change)
			roots = append(roots, s.ID)
		}
	}
	invalidated, err := g.DescendantsIncluding(roots...)
	if err != nil {
		return Impact{}, err
	}
	invalidSet := map[string]struct{}{}
	for _, id := range invalidated {
		invalidSet[id] = struct{}{}
	}
	var unchanged []string
	for _, s := range g.Stages() {
		if _, ok := invalidSet[s.ID]; !ok {
			unchanged = append(unchanged, s.ID)
		}
	}
	return Impact{DirectChanges: direct, InvalidatedStageIDs: invalidated, UnchangedStageIDs: unchanged}, nil
}
