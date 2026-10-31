package publicproduct

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/modelrelease"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

type PreflightModel struct {
	Family                   string    `json:"family"`
	RequiredArchitectureID   string    `json:"required_architecture_id"`
	RequiredChannel          string    `json:"required_channel"`
	ChannelBound             bool      `json:"channel_bound"`
	ModelReleaseID           uuid.UUID `json:"model_release_id,omitempty"`
	ModelArchitectureID      string    `json:"model_architecture_id,omitempty"`
	ModelLifecycleState      string    `json:"model_lifecycle_state,omitempty"`
	ModelEngineReleaseID     uuid.UUID `json:"model_engine_release_id,omitempty"`
	ModelSourceCodeSHA       string    `json:"model_source_code_sha,omitempty"`
	MatchesActiveRelease     bool      `json:"matches_active_release"`
	MatchesRequiredArch      bool      `json:"matches_required_architecture"`
	MatchesCanonicalMainCode bool      `json:"matches_canonical_main_code"`
	Ready                    bool      `json:"ready"`
}

type DeploymentPreflightState struct {
	SchemaVersion               string           `json:"schema_version"`
	Ready                       bool             `json:"ready"`
	ProductionReleaseBound      bool             `json:"production_release_bound"`
	EngineReleaseID             uuid.UUID        `json:"engine_release_id,omitempty"`
	ReleaseChannelLockVersion   int64            `json:"release_channel_lock_version,omitempty"`
	ReleasePurpose              string           `json:"release_purpose,omitempty"`
	CanonicalMainSourceVerified bool             `json:"canonical_main_source_verified"`
	SourceCodeSHA               string           `json:"source_code_sha"`
	Models                      []PreflightModel `json:"models"`
	Blockers                    []string         `json:"blockers"`
	Capabilities                map[string]bool   `json:"capabilities"`
}

// ProductDeploymentPreflightState reads only sealed/registered Platform state.
// It does not create releases, promote models or repair missing bindings.
func ProductDeploymentPreflightState(
	ctx context.Context,
	pool *pgxpool.Pool,
	codeSHA string,
) (DeploymentPreflightState, error) {
	if pool == nil {
		return DeploymentPreflightState{}, fmt.Errorf("pool is required")
	}
	codeSHA = strings.TrimSpace(codeSHA)
	out := DeploymentPreflightState{
		SchemaVersion: "RealSaS.ProductDeploymentPreflight.v1",
		SourceCodeSHA: codeSHA,
		Models:        []PreflightModel{},
		Blockers:      []string{},
		Capabilities: map[string]bool{
			"operator_exact_source_observation_ingress": true,
			"post_proof_workspace_materialization":     true,
			"studio_persistent_authoring":              true,
		},
	}
	if len(codeSHA) != 40 {
		out.Blockers = append(out.Blockers, "SOURCE_CODE_SHA_INVALID")
	}

	channel, err := ActiveProductionRelease(ctx, pool)
	if err != nil {
		if err == ErrReleaseChannelUnbound {
			out.Blockers = append(out.Blockers, "PRODUCTION_RELEASE_CHANNEL_UNBOUND")
			return finalizePreflight(out), nil
		}
		return DeploymentPreflightState{}, err
	}
	out.ProductionReleaseBound = true
	out.EngineReleaseID = channel.EngineReleaseID
	out.ReleaseChannelLockVersion = channel.LockVersion

	purpose, requirements, err := release.LoadModelRequirements(ctx, pool, channel.EngineReleaseID)
	if err != nil {
		return DeploymentPreflightState{}, err
	}
	out.ReleasePurpose = string(purpose)
	if purpose != release.PurposeProduct {
		out.Blockers = append(out.Blockers, "PRODUCTION_RELEASE_PURPOSE_NOT_PRODUCT")
	}
	if len(requirements) == 0 {
		out.Blockers = append(out.Blockers, "PRODUCTION_RELEASE_MODEL_REQUIREMENTS_EMPTY")
	}
	if len(codeSHA) == 40 {
		if err := release.RequireCanonicalMain(ctx, pool, channel.EngineReleaseID, codeSHA); err != nil {
			out.Blockers = append(out.Blockers, "PRODUCTION_RELEASE_CANONICAL_MAIN_SOURCE_MISMATCH")
		} else {
			out.CanonicalMainSourceVerified = true
		}
	}

	channels, err := modelrelease.Channels(ctx, pool)
	if err != nil {
		return DeploymentPreflightState{}, err
	}
	productionByFamily := map[string]modelrelease.ChannelRecord{}
	for _, row := range channels {
		if row.Channel == modelrelease.ChannelProduction {
			productionByFamily[row.Family] = row
		}
	}

	for _, requirement := range requirements {
		model := PreflightModel{
			Family:                 requirement.Family,
			RequiredArchitectureID: requirement.ArchitectureID,
			RequiredChannel:        requirement.Channel,
		}
		if requirement.Channel != string(modelrelease.ChannelProduction) {
			out.Blockers = append(out.Blockers, "MODEL_REQUIREMENT_CHANNEL_NOT_PRODUCTION:"+requirement.Family)
		}
		channelRow, ok := productionByFamily[requirement.Family]
		if !ok {
			out.Blockers = append(out.Blockers, "MODEL_PRODUCTION_CHANNEL_UNBOUND:"+requirement.Family)
			out.Models = append(out.Models, model)
			continue
		}
		model.ChannelBound = true
		model.ModelReleaseID = channelRow.ModelReleaseID
		record, err := modelrelease.Get(ctx, pool, channelRow.ModelReleaseID)
		if err != nil {
			return DeploymentPreflightState{}, err
		}
		model.ModelArchitectureID = record.ArchitectureID
		model.ModelLifecycleState = string(record.LifecycleState)
		model.ModelEngineReleaseID = record.EngineReleaseID
		model.ModelSourceCodeSHA = record.SourceCodeSHA
		model.MatchesRequiredArch = record.ArchitectureID == requirement.ArchitectureID
		model.MatchesActiveRelease = record.EngineReleaseID == channel.EngineReleaseID
		model.MatchesCanonicalMainCode = record.SourceCodeSHA == codeSHA
		model.Ready = model.ChannelBound &&
			record.LifecycleState == modelrelease.StateProduction &&
			model.MatchesRequiredArch &&
			model.MatchesActiveRelease &&
			model.MatchesCanonicalMainCode
		if record.LifecycleState != modelrelease.StateProduction {
			out.Blockers = append(out.Blockers, "MODEL_LIFECYCLE_NOT_PRODUCTION:"+requirement.Family)
		}
		if !model.MatchesRequiredArch {
			out.Blockers = append(out.Blockers, "MODEL_ARCHITECTURE_MISMATCH:"+requirement.Family)
		}
		if !model.MatchesActiveRelease {
			out.Blockers = append(out.Blockers, "MODEL_ENGINE_RELEASE_MISMATCH:"+requirement.Family)
		}
		if !model.MatchesCanonicalMainCode {
			out.Blockers = append(out.Blockers, "MODEL_SOURCE_CODE_MISMATCH:"+requirement.Family)
		}
		out.Models = append(out.Models, model)
	}
	return finalizePreflight(out), nil
}

func finalizePreflight(out DeploymentPreflightState) DeploymentPreflightState {
	sort.Slice(out.Models, func(i, j int) bool { return out.Models[i].Family < out.Models[j].Family })
	sort.Strings(out.Blockers)
	out.Blockers = uniqueStrings(out.Blockers)
	allModels := len(out.Models) > 0
	for _, model := range out.Models {
		allModels = allModels && model.Ready
	}
	out.Ready = out.ProductionReleaseBound &&
		out.ReleasePurpose == string(release.PurposeProduct) &&
		out.CanonicalMainSourceVerified &&
		allModels && len(out.Blockers) == 0
	return out
}

func uniqueStrings(values []string) []string {
	if len(values) < 2 {
		return values
	}
	out := values[:0]
	for _, value := range values {
		if len(out) == 0 || out[len(out)-1] != value {
			out = append(out, value)
		}
	}
	return out
}
