package release

import (
	"fmt"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type StageVersion struct {
	GraphNodeSHA256      string
	ImplementationSHA256 string
	PolicySHA256         string
	ParametersSHA256     string
}

type DirectChange struct {
	StageID               string
	ImplementationChanged bool
	PolicyChanged         bool
	ParametersChanged     bool
	GraphChanged          bool
	Added                 bool
	Removed               bool
}

type Impact struct {
	DirectChanges       []DirectChange
	InvalidatedStageIDs []string
	UnchangedStageIDs   []string
	RemovedStageIDs     []string
}

func Compare(g *stagegraph.Graph, baseline, candidate map[string]StageVersion) (Impact, error) {
	return CompareGraphs(g, g, baseline, candidate)
}

func CompareGraphs(oldGraph, g *stagegraph.Graph, baseline, candidate map[string]StageVersion) (Impact, error) {
	var direct []DirectChange
	var roots []string
	var removed []string
	for _, old := range oldGraph.Stages() {
		if _, ok := baseline[old.ID]; !ok {
			return Impact{}, fmt.Errorf("missing baseline stage version %s", old.ID)
		}
		if _, ok := g.Get(old.ID); !ok {
			removed = append(removed, old.ID)
			direct = append(direct, DirectChange{StageID: old.ID, Removed: true})
		}
	}
	for _, s := range g.Stages() {
		a, okA := baseline[s.ID]
		b, okB := candidate[s.ID]
		if !okB {
			return Impact{}, fmt.Errorf("missing stage version %s", s.ID)
		}
		old, existed := oldGraph.Get(s.ID)
		if existed && !okA {
			return Impact{}, fmt.Errorf("missing baseline stage version %s", s.ID)
		}
		oldNodeSHA := ""
		if existed {
			var err error
			oldNodeSHA, err = old.NodeSHA256()
			if err != nil {
				return Impact{}, err
			}
		}
		newNodeSHA, err := s.NodeSHA256()
		if err != nil {
			return Impact{}, err
		}
		change := DirectChange{
			StageID:               s.ID,
			Added:                 !existed,
			GraphChanged:          existed && oldNodeSHA != newNodeSHA,
			ImplementationChanged: a.ImplementationSHA256 != b.ImplementationSHA256,
			PolicyChanged:         a.PolicySHA256 != b.PolicySHA256,
			ParametersChanged:     a.ParametersSHA256 != b.ParametersSHA256,
		}
		if change.Added || change.GraphChanged || change.ImplementationChanged || change.PolicyChanged || change.ParametersChanged {
			direct = append(direct, change)
			roots = append(roots, s.ID)
		}
	}
	if len(removed) > 0 {
		oldAffected, err := oldGraph.DescendantsIncluding(removed...)
		if err != nil {
			return Impact{}, err
		}
		for _, id := range oldAffected {
			if _, survives := g.Get(id); survives {
				roots = append(roots, id)
			}
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
	return Impact{DirectChanges: direct, InvalidatedStageIDs: invalidated, UnchangedStageIDs: unchanged, RemovedStageIDs: removed}, nil
}
