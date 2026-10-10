package architecture

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type Authority string

const (
	AuthorityPlatform Authority = "PLATFORM"
	AuthorityEngine   Authority = "ENGINE"
	AuthorityRuntime  Authority = "RUNTIME"
)

type Module struct {
	ID               string    `json:"id"`
	ParentDomain     string    `json:"parent_domain"`
	Authority        Authority `json:"authority"`
	Purpose          string    `json:"purpose"`
	CodeRoots        []string  `json:"code_roots"`
	StateTables      []string  `json:"state_tables,omitempty"`
	ArtifactFamilies []string  `json:"artifact_families,omitempty"`
	FailurePrefixes  []string  `json:"failure_prefixes,omitempty"`
	UpstreamModules  []string  `json:"upstream_modules,omitempty"`
}

type Domain struct {
	ID        string    `json:"id"`
	Authority Authority `json:"authority"`
	Purpose   string    `json:"purpose"`
	Modules   []Module  `json:"modules"`
}

type StageOwner struct {
	Ordinal         int      `json:"ordinal"`
	StageID         string   `json:"stage_id"`
	Group           string   `json:"group"`
	OwnerModuleID   string   `json:"owner_module_id"`
	OwnershipSource string   `json:"ownership_source"`
	Adapter         string   `json:"adapter"`
	DependsOn       []string `json:"depends_on"`
}

type Registry struct {
	ContractVersion string       `json:"contract_version"`
	SystemID        string       `json:"system_id"`
	Domains         []Domain     `json:"domains"`
	Stages          []StageOwner `json:"stages"`
}

var engineGroupOwner = map[string]string{
	"source":              "engine.source",
	"observation":         "engine.observation",
	"geometry":            "engine.geometry",
	"rig":                 "engine.rig",
	"presentation_domain": "engine.presentation_domain",
	"components":          "engine.components",
	"mesh":                "engine.mesh",
	"appearance":          "engine.appearance",
	"skin":                "engine.skin",
	"presentation":        "engine.presentation",
	"product_state":       "engine.product_state",
	"motion":              "engine.motion",
	"runtime":             "engine.runtime_binding",
	"closure":             "engine.closure",
}

func staticDomains() []Domain {
	return []Domain{
		{
			ID:        "platform",
			Authority: AuthorityPlatform,
			Purpose:   "Durable product/research state, orchestration, transactions, registry metadata and query surfaces.",
			Modules: []Module{
				{
					ID: "platform.infrastructure", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Control-plane boot, migrations, persistence primitives, semantic hashing, architecture introspection and canonical stage-graph loading.",
					CodeRoots:       []string{"platform/cmd", "platform/migrations", "platform/internal/persistence", "platform/internal/semantic", "platform/internal/stagegraph", "platform/internal/architecture"},
					FailurePrefixes: []string{"PLATFORM_", "MIGRATION_", "PERSISTENCE_", "ARCHITECTURE_"},
				},
				{
					ID: "platform.artifact", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:          "Immutable artifact metadata, CAS transport and immutable subject input sets.",
					CodeRoots:        []string{"platform/internal/artifactstore", "platform/internal/registry", "platform/internal/input"},
					StateTables:      []string{"artifact_types", "artifacts", "artifact_inputs", "subject_inputs", "subject_input_artifacts"},
					ArtifactFamilies: []string{"RealSaS.StageResultManifest", "RealSaS.SubjectInputManifest"},
					FailurePrefixes:  []string{"ARTIFACT_", "CAS_", "SUBJECT_INPUT_"},
				},
				{
					ID: "platform.release", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Immutable engine-graph release snapshots and semantic version identity.",
					CodeRoots:       []string{"platform/internal/release"},
					StateTables:     []string{"engine_releases", "engine_release_stages"},
					FailurePrefixes: []string{"ENGINE_RELEASE_"},
				},
				{
					ID: "platform.research", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Research Attempt lineage, code-change impact and repair continuation.",
					CodeRoots:       []string{"platform/internal/attempt", "platform/internal/diagnostic", "platform/internal/agentsession"},
					StateTables:     []string{"attempts", "attempt_events", "attempt_artifacts", "failure_signatures", "owner_attributions", "repair_directives"},
					FailurePrefixes: []string{"RESEARCH_", "AGENT_", "ILLEGAL_FAILURE_OWNER", "REPAIR_"},
					UpstreamModules: []string{"platform.release", "platform.workflow"},
				},
				{
					ID: "platform.workflow", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Commands, outbox delivery, Temporal workflows, stage execution state and crash-safe resume.",
					CodeRoots:       []string{"platform/internal/command", "platform/internal/httpapi", "platform/internal/outbox", "platform/internal/dispatch", "platform/internal/orchestration", "platform/internal/control", "platform/internal/capability", "platform/internal/resolver"},
					StateTables:     []string{"commands", "outbox_events", "executions", "execution_artifacts", "compiler_run_bindings"},
					FailurePrefixes: []string{"WORKFLOW_", "EXECUTION_", "COMPILER_PLATFORM_PLAN_DRIFT"},
					UpstreamModules: []string{"platform.artifact", "platform.release"},
				},
				{
					ID: "platform.proof", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Persistent proof/qualification indexing used for reuse and promotion decisions.",
					CodeRoots:       []string{"platform/internal/registry", "platform/internal/diagnostic"},
					StateTables:     []string{"proofs", "qualifications"},
					FailurePrefixes: []string{"PROOF_", "QUALIFICATION_"},
				},
				{
					ID: "platform.product", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "ProductRevision sealing, atomic promotion and current production truth.",
					CodeRoots:       []string{"platform/internal/product"},
					StateTables:     []string{"product_revisions", "product_revision_artifacts", "promotions", "subject_current_revision"},
					FailurePrefixes: []string{"PROMOTION_", "PRODUCT_REVISION_"},
					UpstreamModules: []string{"platform.artifact", "platform.proof"},
				},
				{
					ID: "platform.render", ParentDomain: "platform", Authority: AuthorityPlatform,
					Purpose:         "Exact ProductRevision-bound render requests, render cache identity and output registration.",
					CodeRoots:       []string{"platform/internal/orchestration", "platform/internal/command"},
					StateTables:     []string{"render_requests", "render_outputs"},
					FailurePrefixes: []string{"RENDER_"},
					UpstreamModules: []string{"platform.product", "platform.artifact", "platform.workflow"},
				},
			},
		},
		{
			ID:        "engine",
			Authority: AuthorityEngine,
			Purpose:   "Scientific compiler meaning: stage contracts, adapters, models, geometry/mechanics/appearance/motion and proof evidence.",
			Modules: []Module{
				engineModule("source", "Source bytes/provenance/mechanical admission.", "compiler/realsas_compiler_services/orchestrator/adapters/source.py"),
				engineModule("observation", "Observation/camera/admission contracts.", "compiler/realsas_compiler_services/orchestrator/adapters/observation_v2.py"),
				engineModule("geometry", "IRIS, zero-surface, geometry substrate and GSA.", "compiler/realsas_compiler_services/orchestrator/adapters/iris_geometry_v2.py"),
				engineModule("rig", "Rigging substrate, Geppetto and skeleton qualification.", "compiler/realsas_compiler_services/orchestrator/adapters/learned_mechanics_v2.py"),
				engineModule("presentation_domain", "Output presentation direction authority.", "compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py"),
				engineModule("components", "Mechanical partition/component carrier authority.", "compiler/realsas_compiler_services/orchestrator/adapters/mesh_v2.py"),
				engineModule("mesh", "Canonical mesh addressing, deformation envelope and mechanical mesh.", "compiler/realsas_compiler_services/orchestrator/adapters/mesh_v2.py"),
				engineModule("appearance", "CAA compile, source-owned appearance baking and appearance proof.", "compiler/realsas_compiler_services/orchestrator/adapters/appearance_v2.py"),
				engineModule("skin", "Arachne and skin qualification.", "compiler/realsas_compiler_services/orchestrator/adapters/learned_mechanics_v2.py"),
				engineModule("presentation", "Final qualified presentation structure authority.", "compiler/realsas_compiler_services/orchestrator/adapters/product_state_v2.py"),
				engineModule("product_state", "Canonical puppet seal inside compiler semantics.", "compiler/realsas_compiler_services/orchestrator/adapters/product_state_v2.py"),
				engineModule("motion", "Motion source, compile and dynamic proof.", "compiler/realsas_compiler_services/orchestrator/adapters/motion_v2.py"),
				engineModule("runtime_binding", "Runtime projection, package materialization and runtime visual proof.", "compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py"),
				engineModule("closure", "Final compiler product-closure authority.", "compiler/realsas_compiler_services/orchestrator/adapters/closure_v2.py"),
			},
		},
		{
			ID:        "runtime",
			Authority: AuthorityRuntime,
			Purpose:   "Native package consumption, deformation/playback and rendering hot paths.",
			Modules: []Module{
				{
					ID: "runtime.package", ParentDomain: "runtime", Authority: AuthorityRuntime,
					Purpose:         "Open and validate sealed .rss/runtime packages.",
					CodeRoots:       []string{"runtime/realsas_cpp"},
					FailurePrefixes: []string{"NATIVE_PACKAGE_", "RUNTIME_PACKAGE_"},
				},
				{
					ID: "runtime.deformation", ParentDomain: "runtime", Authority: AuthorityRuntime,
					Purpose:         "Native deformation, skinning and motion playback.",
					CodeRoots:       []string{"runtime/realsas_cpp"},
					FailurePrefixes: []string{"DEFORMATION_", "PLAYBACK_"},
					UpstreamModules: []string{"runtime.package"},
				},
				{
					ID: "runtime.render", ParentDomain: "runtime", Authority: AuthorityRuntime,
					Purpose:         "Native source-owned visual rendering and frame output.",
					CodeRoots:       []string{"runtime/realsas_cpp"},
					FailurePrefixes: []string{"RENDER_", "VISUAL_INTEGRITY_"},
					UpstreamModules: []string{"runtime.package", "runtime.deformation"},
				},
			},
		},
	}
}

func normalizedGroupID(group string) string {
	group = strings.ToLower(strings.TrimSpace(group))
	if group == "" {
		return "unclassified"
	}
	var b strings.Builder
	lastUnderscore := false
	for _, r := range group {
		valid := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if valid {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "unclassified"
	}
	return out
}

func adapterCodeRoot(adapter string) string {
	module := strings.SplitN(adapter, ":", 2)[0]
	if module == "" {
		return "compiler"
	}
	return strings.ReplaceAll(module, ".", "/") + ".py"
}

func engineModule(group, purpose, codeRoot string) Module {
	id := "engine." + group
	return Module{
		ID: id, ParentDomain: "engine", Authority: AuthorityEngine,
		Purpose:          purpose,
		CodeRoots:        []string{codeRoot},
		ArtifactFamilies: []string{"RealSaS.StageResultManifest"},
		FailurePrefixes:  []string{strings.ToUpper(group) + "_", "STAGE_"},
	}
}

func Build(g *stagegraph.Graph) (Registry, error) {
	if g == nil {
		return Registry{}, errors.New("stage graph is required")
	}
	domains := staticDomains()
	moduleIDs := map[string]struct{}{}
	for _, d := range domains {
		for _, m := range d.Modules {
			if _, exists := moduleIDs[m.ID]; exists {
				return Registry{}, fmt.Errorf("duplicate architecture module %s", m.ID)
			}
			moduleIDs[m.ID] = struct{}{}
		}
	}
	stages := make([]StageOwner, 0, g.StageCount())
	for _, s := range g.Stages() {
		owner, ok := engineGroupOwner[s.Group]
		ownershipSource := "DECLARED_GROUP_MAPPING"
		if !ok {
			owner = "engine." + normalizedGroupID(s.Group)
			ownershipSource = "DISCOVERED_STAGE_GROUP"
		}
		if _, exists := moduleIDs[owner]; !exists {
			engineIndex := -1
			for i := range domains {
				if domains[i].ID == "engine" {
					engineIndex = i
					break
				}
			}
			if engineIndex < 0 {
				return Registry{}, errors.New("engine domain missing")
			}
			dynamic := Module{
				ID: owner, ParentDomain: "engine", Authority: AuthorityEngine,
				Purpose:          "Dynamically discovered compiler stage group " + s.Group + ".",
				CodeRoots:        []string{adapterCodeRoot(s.Adapter)},
				ArtifactFamilies: []string{"RealSaS.StageResultManifest"},
				FailurePrefixes:  []string{strings.ToUpper(normalizedGroupID(s.Group)) + "_", "STAGE_"},
			}
			domains[engineIndex].Modules = append(domains[engineIndex].Modules, dynamic)
			moduleIDs[owner] = struct{}{}
		}
		stages = append(stages, StageOwner{
			Ordinal: s.Ordinal, StageID: s.ID, Group: s.Group, OwnerModuleID: owner,
			OwnershipSource: ownershipSource,
			Adapter:         s.Adapter, DependsOn: append([]string(nil), s.DependsOn...),
		})
	}
	if len(stages) != g.StageCount() {
		return Registry{}, fmt.Errorf("architecture registry stage count drift: registry=%d graph=%d", len(stages), g.StageCount())
	}
	return Registry{
		ContractVersion: "RealSaS.SystemArchitectureRegistry.v1",
		SystemID:        "realsas",
		Domains:         domains,
		Stages:          stages,
	}, nil
}

func (r Registry) Module(id string) (Module, bool) {
	for _, d := range r.Domains {
		for _, m := range d.Modules {
			if m.ID == id {
				return m, true
			}
		}
	}
	return Module{}, false
}

func (r Registry) Stage(stageID string) (StageOwner, bool) {
	for _, s := range r.Stages {
		if s.StageID == stageID {
			return s, true
		}
	}
	return StageOwner{}, false
}

func (r Registry) Validate() error {
	if r.ContractVersion == "" || r.SystemID != "realsas" {
		return errors.New("invalid architecture registry identity")
	}
	seenModules := map[string]struct{}{}
	for _, d := range r.Domains {
		for _, m := range d.Modules {
			if m.ParentDomain != d.ID || m.ID == "" || len(m.CodeRoots) == 0 {
				return fmt.Errorf("invalid module contract %s", m.ID)
			}
			seenModules[m.ID] = struct{}{}
		}
	}
	if len(r.Stages) == 0 {
		return fmt.Errorf("architecture registry must own at least one compiler stage")
	}
	for _, s := range r.Stages {
		if _, ok := seenModules[s.OwnerModuleID]; !ok {
			return fmt.Errorf("stage %s references unknown owner %s", s.StageID, s.OwnerModuleID)
		}
	}
	return nil
}

func (r Registry) ModuleIDs() []string {
	var ids []string
	for _, d := range r.Domains {
		for _, m := range d.Modules {
			ids = append(ids, m.ID)
		}
	}
	sort.Strings(ids)
	return ids
}
