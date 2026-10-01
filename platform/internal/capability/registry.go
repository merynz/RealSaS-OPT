package capability

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type Kind string

const (
	KindStage   Kind = "STAGE"
	KindModel   Kind = "MODEL"
	KindModule  Kind = "MODULE"
	KindTool    Kind = "TOOL"
	KindProof   Kind = "PROOF"
	KindRuntime Kind = "RUNTIME"
)

type Mode string

const (
	ModeProduct   Mode = "PRODUCT"
	ModeDeveloper Mode = "DEVELOPER"
)

type Descriptor struct {
	ID                   string         `json:"id"`
	Kind                 Kind           `json:"kind"`
	OwnerModuleID        string         `json:"owner_module_id"`
	ExecutorActivity     string         `json:"executor_activity"`
	Dependencies         []string       `json:"dependencies"`
	AllowedModes         []Mode         `json:"allowed_modes"`
	ProductPassAuthority bool           `json:"product_pass_authority"`
	Aliases              []string       `json:"aliases,omitempty"`
	Metadata             map[string]any `json:"metadata,omitempty"`
}

func (d Descriptor) Validate() error {
	if strings.TrimSpace(d.ID) == "" {
		return errors.New("capability id is required")
	}
	switch d.Kind {
	case KindStage, KindModel, KindModule, KindTool, KindProof, KindRuntime:
	default:
		return fmt.Errorf("capability %s has invalid kind %q", d.ID, d.Kind)
	}
	if d.OwnerModuleID == "" || d.ExecutorActivity == "" {
		return fmt.Errorf("capability %s requires owner module and executor activity", d.ID)
	}
	if len(d.AllowedModes) == 0 {
		return fmt.Errorf("capability %s has no allowed execution modes", d.ID)
	}
	seenMode := map[Mode]struct{}{}
	for _, mode := range d.AllowedModes {
		if mode != ModeProduct && mode != ModeDeveloper {
			return fmt.Errorf("capability %s has invalid mode %q", d.ID, mode)
		}
		if _, ok := seenMode[mode]; ok {
			return fmt.Errorf("capability %s duplicates mode %q", d.ID, mode)
		}
		seenMode[mode] = struct{}{}
	}
	if d.Metadata == nil {
		d.Metadata = map[string]any{}
	}
	return nil
}

type Provider interface {
	Capabilities(context.Context) ([]Descriptor, error)
}

type ProviderFunc func(context.Context) ([]Descriptor, error)

func (fn ProviderFunc) Capabilities(ctx context.Context) ([]Descriptor, error) {
	return fn(ctx)
}

type Registry struct {
	ordered []Descriptor
	byID    map[string]Descriptor
}

func Build(ctx context.Context, providers ...Provider) (*Registry, error) {
	var descriptors []Descriptor
	for _, provider := range providers {
		if provider == nil {
			return nil, errors.New("nil capability provider")
		}
		rows, err := provider.Capabilities(ctx)
		if err != nil {
			return nil, err
		}
		descriptors = append(descriptors, rows...)
	}
	return NewRegistry(descriptors)
}

func NewRegistry(descriptors []Descriptor) (*Registry, error) {
	if len(descriptors) == 0 {
		return nil, errors.New("capability registry cannot be empty")
	}
	byID := make(map[string]Descriptor, len(descriptors))
	for _, descriptor := range descriptors {
		if err := descriptor.Validate(); err != nil {
			return nil, err
		}
		if _, exists := byID[descriptor.ID]; exists {
			return nil, fmt.Errorf("duplicate capability id %s", descriptor.ID)
		}
		copy := descriptor
		copy.Dependencies = append([]string(nil), descriptor.Dependencies...)
		copy.AllowedModes = append([]Mode(nil), descriptor.AllowedModes...)
		copy.Aliases = append([]string(nil), descriptor.Aliases...)
		if copy.Metadata == nil {
			copy.Metadata = map[string]any{}
		}
		byID[copy.ID] = copy
	}
	for _, descriptor := range byID {
		for _, dep := range descriptor.Dependencies {
			if _, ok := byID[dep]; !ok {
				return nil, fmt.Errorf("capability %s depends on unknown capability %s", descriptor.ID, dep)
			}
		}
	}
	ordered, err := topologicalOrder(byID)
	if err != nil {
		return nil, err
	}
	return &Registry{ordered: ordered, byID: byID}, nil
}

func topologicalOrder(byID map[string]Descriptor) ([]Descriptor, error) {
	indegree := make(map[string]int, len(byID))
	children := make(map[string][]string, len(byID))
	for id, descriptor := range byID {
		indegree[id] = len(descriptor.Dependencies)
		for _, dep := range descriptor.Dependencies {
			children[dep] = append(children[dep], id)
		}
	}
	var ready []string
	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}
	sort.Strings(ready)
	var out []Descriptor
	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]
		out = append(out, byID[id])
		for _, child := range children[id] {
			indegree[child]--
			if indegree[child] == 0 {
				ready = append(ready, child)
				sort.Strings(ready)
			}
		}
	}
	if len(out) != len(byID) {
		var cyclic []string
		for id, degree := range indegree {
			if degree > 0 {
				cyclic = append(cyclic, id)
			}
		}
		sort.Strings(cyclic)
		return nil, fmt.Errorf("capability dependency cycle: %v", cyclic)
	}
	return out, nil
}

func (r *Registry) Descriptors() []Descriptor {
	return append([]Descriptor(nil), r.ordered...)
}

func (r *Registry) Get(id string) (Descriptor, bool) {
	d, ok := r.byID[id]
	return d, ok
}

func (r *Registry) Resolve(targets ...string) ([]Descriptor, error) {
	if len(targets) == 0 {
		return nil, errors.New("at least one target capability is required")
	}
	required := map[string]struct{}{}
	var visit func(string) error
	visit = func(id string) error {
		descriptor, ok := r.byID[id]
		if !ok {
			return fmt.Errorf("unknown target capability %s", id)
		}
		if _, ok := required[id]; ok {
			return nil
		}
		required[id] = struct{}{}
		for _, dep := range descriptor.Dependencies {
			if err := visit(dep); err != nil {
				return err
			}
		}
		return nil
	}
	for _, target := range targets {
		if err := visit(target); err != nil {
			return nil, err
		}
	}
	out := make([]Descriptor, 0, len(required))
	for _, descriptor := range r.ordered {
		if _, ok := required[descriptor.ID]; ok {
			out = append(out, descriptor)
		}
	}
	return out, nil
}

func (r *Registry) Search(term string) []Descriptor {
	needle := strings.ToLower(strings.TrimSpace(term))
	if needle == "" {
		return r.Descriptors()
	}
	var out []Descriptor
	for _, descriptor := range r.ordered {
		hay := []string{descriptor.ID, string(descriptor.Kind), descriptor.OwnerModuleID, descriptor.ExecutorActivity}
		hay = append(hay, descriptor.Aliases...)
		if strings.Contains(strings.ToLower(strings.Join(hay, " ")), needle) {
			out = append(out, descriptor)
		}
	}
	return out
}

type CompilePlanProvider struct {
	Graph        *stagegraph.Graph
	Architecture architecture.Registry
}

func (p CompilePlanProvider) Capabilities(_ context.Context) ([]Descriptor, error) {
	if p.Graph == nil {
		return nil, errors.New("compile plan provider requires graph")
	}
	rows := make([]Descriptor, 0, p.Graph.StageCount())
	for _, stage := range p.Graph.Stages() {
		owner, ok := p.Architecture.Stage(stage.ID)
		if !ok {
			return nil, fmt.Errorf("stage %s has no architecture owner", stage.ID)
		}
		deps := make([]string, 0, len(stage.DependsOn))
		for _, dep := range stage.DependsOn {
			deps = append(deps, StageCapabilityID(dep))
		}
		aliases := []string{stage.ID, stage.Group, stage.Adapter}
		rows = append(rows, Descriptor{
			ID:                   StageCapabilityID(stage.ID),
			Kind:                 KindStage,
			OwnerModuleID:        owner.OwnerModuleID,
			ExecutorActivity:     "engine.execute_compile_stage.v1",
			Dependencies:         deps,
			AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
			ProductPassAuthority: stage.Policy.ProductPassAuthority,
			Aliases:              aliases,
			Metadata: map[string]any{
				"stage_id":         stage.ID,
				"stage_ordinal":    stage.Ordinal,
				"stage_group":      stage.Group,
				"adapter_ref":      stage.Adapter,
				"ownership_source": owner.OwnershipSource,
			},
		})
	}
	return rows, nil
}

func StageCapabilityID(stageID string) string {
	return "compiler.stage/" + stageID
}
