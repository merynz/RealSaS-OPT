package stagegraph

import (
	"encoding/json"
	"errors"
	"fmt"
)

type Policy struct {
	FailClosed           bool `json:"fail_closed"`
	Cacheable            bool `json:"cacheable"`
	OutputHashRequired   bool `json:"output_hash_required"`
	ProductPassAuthority bool `json:"product_pass_authority"`
}

type Stage struct {
	Ordinal   int      `json:"ordinal"`
	ID        string   `json:"id"`
	Title     string   `json:"title"`
	Group     string   `json:"group"`
	DependsOn []string `json:"depends_on"`
	Adapter   string   `json:"adapter"`
	Policy    Policy   `json:"policy"`
}

type planFile struct {
	StageCount int     `json:"stage_count"`
	Stages     []Stage `json:"stages"`
}

type Graph struct {
	stages             []Stage
	byID               map[string]Stage
	children           map[string][]string
	productPassStageID string
}

func ParsePlan(data []byte, requireProductPassAuthority bool) (*Graph, error) {
	var p planFile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	if len(p.Stages) == 0 {
		return nil, errors.New("execution plan must contain at least one stage")
	}
	if p.StageCount != len(p.Stages) {
		return nil, fmt.Errorf("declared stage_count=%d does not match stages=%d", p.StageCount, len(p.Stages))
	}
	byID := make(map[string]Stage, len(p.Stages))
	children := make(map[string][]string, len(p.Stages))
	seen := make(map[string]struct{}, len(p.Stages))
	var passOwners []string
	for i, s := range p.Stages {
		if s.Ordinal != i+1 || s.ID == "" {
			return nil, fmt.Errorf("stage ordinal/id drift at index %d", i)
		}
		if _, ok := seen[s.ID]; ok {
			return nil, fmt.Errorf("duplicate stage id %s", s.ID)
		}
		for _, dep := range s.DependsOn {
			if _, ok := seen[dep]; !ok {
				return nil, fmt.Errorf("%s depends on future/unknown stage %s", s.ID, dep)
			}
			children[dep] = append(children[dep], s.ID)
		}
		if s.Policy.ProductPassAuthority {
			passOwners = append(passOwners, s.ID)
		}
		seen[s.ID] = struct{}{}
		byID[s.ID] = s
		children[s.ID] = append([]string(nil), children[s.ID]...)
	}
	if len(passOwners) > 1 {
		return nil, fmt.Errorf("multiple product-pass authorities: %v", passOwners)
	}
	if requireProductPassAuthority && len(passOwners) != 1 {
		return nil, errors.New("canonical product plan requires exactly one product-pass authority")
	}
	productPassStageID := ""
	if len(passOwners) == 1 {
		productPassStageID = passOwners[0]
	}
	return &Graph{
		stages:             append([]Stage(nil), p.Stages...),
		byID:               byID,
		children:           children,
		productPassStageID: productPassStageID,
	}, nil
}

func ParseCanonicalPlan(data []byte) (*Graph, error) {
	return ParsePlan(data, true)
}

func (g *Graph) Stages() []Stage { return append([]Stage(nil), g.stages...) }
func (g *Graph) StageCount() int { return len(g.stages) }

func (g *Graph) ProductPassStageID() (string, bool) {
	if g.productPassStageID == "" {
		return "", false
	}
	return g.productPassStageID, true
}

func (g *Graph) Get(id string) (Stage, bool) {
	s, ok := g.byID[id]
	return s, ok
}

func (g *Graph) DescendantsIncluding(roots ...string) ([]string, error) {
	affected := map[string]struct{}{}
	pending := append([]string(nil), roots...)
	for _, root := range roots {
		if _, ok := g.byID[root]; !ok {
			return nil, fmt.Errorf("unknown stage %s", root)
		}
		affected[root] = struct{}{}
	}
	for len(pending) > 0 {
		cur := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		for _, child := range g.children[cur] {
			if _, ok := affected[child]; ok {
				continue
			}
			affected[child] = struct{}{}
			pending = append(pending, child)
		}
	}
	out := make([]string, 0, len(affected))
	for _, s := range g.stages {
		if _, ok := affected[s.ID]; ok {
			out = append(out, s.ID)
		}
	}
	return out, nil
}

func (g *Graph) AncestorsIncluding(targets ...string) ([]string, error) {
	required := map[string]struct{}{}
	pending := append([]string(nil), targets...)
	for _, target := range targets {
		if _, ok := g.byID[target]; !ok {
			return nil, fmt.Errorf("unknown stage %s", target)
		}
		required[target] = struct{}{}
	}
	for len(pending) > 0 {
		cur := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		s := g.byID[cur]
		for _, parent := range s.DependsOn {
			if _, ok := required[parent]; ok {
				continue
			}
			required[parent] = struct{}{}
			pending = append(pending, parent)
		}
	}
	out := make([]string, 0, len(required))
	for _, s := range g.stages {
		if _, ok := required[s.ID]; ok {
			out = append(out, s.ID)
		}
	}
	return out, nil
}
