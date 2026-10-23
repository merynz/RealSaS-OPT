package operational

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/experiment"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelrelease"
)

type AttemptStatus struct {
	ID                   uuid.UUID  `json:"id"`
	SubjectID            uuid.UUID  `json:"subject_id"`
	ExperimentID         *uuid.UUID `json:"experiment_id,omitempty"`
	EngineReleaseID      *uuid.UUID `json:"engine_release_id,omitempty"`
	EngineReleasePurpose *string    `json:"engine_release_purpose,omitempty"`
	EngineReleaseSHA256  *string    `json:"engine_release_sha256,omitempty"`
	CapabilitySetSHA256  *string    `json:"capability_set_sha256,omitempty"`
	Kind                 string     `json:"kind"`
	FinalState           string     `json:"final_state"`
	CreatedAt            time.Time  `json:"created_at"`
}

type CampaignStatus struct {
	ID           uuid.UUID  `json:"id"`
	Name         string     `json:"name"`
	ExperimentID *uuid.UUID `json:"experiment_id,omitempty"`
	State        string     `json:"state"`
	TargetStage  string     `json:"target_stage_id"`
	CreatedAt    time.Time  `json:"created_at"`
}

type CurrentModel struct {
	Channel modelrelease.ChannelRecord `json:"channel"`
	Release modelrelease.Record        `json:"release"`
}

type ProductModelBinding struct {
	Family                  string             `json:"family"`
	ModelReleaseID          uuid.UUID          `json:"model_release_id"`
	ArchitectureID          string             `json:"architecture_id"`
	CheckpointArtifactID    uuid.UUID          `json:"checkpoint_artifact_id"`
	CheckpointContentSHA256 string             `json:"checkpoint_content_sha256"`
	LifecycleState          modelrelease.State `json:"lifecycle_state"`
}

type ProductArtifactBinding struct {
	Role           string    `json:"role"`
	ArtifactID     uuid.UUID `json:"artifact_id"`
	ArtifactType   string    `json:"artifact_type"`
	SchemaVersion  string    `json:"schema_version"`
	SemanticSHA256 string    `json:"semantic_sha256"`
	ContentSHA256  string    `json:"content_sha256"`
}

type CurrentProduct struct {
	SubjectID           uuid.UUID                `json:"subject_id"`
	SubjectSlug         string                   `json:"subject_slug"`
	ProductRevisionID   uuid.UUID                `json:"product_revision_id"`
	RevisionNumber      int64                    `json:"revision_number"`
	ManifestSHA256      string                   `json:"manifest_sha256"`
	EngineReleaseID     *uuid.UUID               `json:"engine_release_id,omitempty"`
	EngineReleaseSHA256 *string                  `json:"engine_release_sha256,omitempty"`
	CapabilitySetSHA256 *string                  `json:"capability_set_sha256,omitempty"`
	Models              []ProductModelBinding    `json:"models"`
	Artifacts           []ProductArtifactBinding `json:"artifacts"`
	UpdatedAt           time.Time                `json:"updated_at"`
}

type Snapshot struct {
	Schema            string              `json:"schema"`
	GeneratedAt       time.Time           `json:"generated_at"`
	ActiveExperiments []experiment.Record `json:"active_experiments"`
	OpenAttempts      []AttemptStatus     `json:"open_attempts"`
	ActiveCampaigns   []CampaignStatus    `json:"active_campaigns"`
	CurrentModels     []CurrentModel      `json:"current_models"`
	CurrentProducts   []CurrentProduct    `json:"current_products"`
}

func Status(ctx context.Context, pool *pgxpool.Pool) (Snapshot, error) {
	out := Snapshot{Schema: "RealSaS.OperationalStatus.v3", GeneratedAt: time.Now().UTC()}

	experiments, err := experiment.List(ctx, pool)
	if err != nil {
		return Snapshot{}, err
	}
	for _, record := range experiments {
		if record.LifecycleState == experiment.StatePreregistered || record.LifecycleState == experiment.StateRunning {
			out.ActiveExperiments = append(out.ActiveExperiments, record)
		}
	}

	attemptRows, err := pool.Query(ctx, `
		SELECT a.id,a.subject_id,a.experiment_id,a.engine_release_id,
		       er.purpose,er.release_sha256,er.capability_set_sha256,
		       a.kind,a.final_state,a.created_at
		FROM attempts a
		LEFT JOIN engine_releases er ON er.id=a.engine_release_id
		WHERE a.final_state='OPEN'
		ORDER BY a.created_at,a.id
	`)
	if err != nil {
		return Snapshot{}, err
	}
	for attemptRows.Next() {
		var record AttemptStatus
		if err := attemptRows.Scan(
			&record.ID,
			&record.SubjectID,
			&record.ExperimentID,
			&record.EngineReleaseID,
			&record.EngineReleasePurpose,
			&record.EngineReleaseSHA256,
			&record.CapabilitySetSHA256,
			&record.Kind,
			&record.FinalState,
			&record.CreatedAt,
		); err != nil {
			attemptRows.Close()
			return Snapshot{}, err
		}
		out.OpenAttempts = append(out.OpenAttempts, record)
	}
	attemptRows.Close()
	if err := attemptRows.Err(); err != nil {
		return Snapshot{}, err
	}

	campaignRows, err := pool.Query(ctx, `
		SELECT id,name,experiment_id,state,target_stage_id,created_at
		FROM research_campaigns
		WHERE state IN ('STARTING','READY','RUNNING')
		ORDER BY created_at,id
	`)
	if err != nil {
		return Snapshot{}, err
	}
	for campaignRows.Next() {
		var record CampaignStatus
		if err := campaignRows.Scan(&record.ID, &record.Name, &record.ExperimentID, &record.State, &record.TargetStage, &record.CreatedAt); err != nil {
			campaignRows.Close()
			return Snapshot{}, err
		}
		out.ActiveCampaigns = append(out.ActiveCampaigns, record)
	}
	campaignRows.Close()
	if err := campaignRows.Err(); err != nil {
		return Snapshot{}, err
	}

	channels, err := modelrelease.Channels(ctx, pool)
	if err != nil {
		return Snapshot{}, err
	}
	for _, channel := range channels {
		release, err := modelrelease.Get(ctx, pool, channel.ModelReleaseID)
		if err != nil {
			return Snapshot{}, err
		}
		out.CurrentModels = append(out.CurrentModels, CurrentModel{Channel: channel, Release: release})
	}

	products, err := loadCurrentProducts(ctx, pool, nil)
	if err != nil {
		return Snapshot{}, err
	}
	out.CurrentProducts = products

	if out.ActiveExperiments == nil {
		out.ActiveExperiments = []experiment.Record{}
	}
	if out.OpenAttempts == nil {
		out.OpenAttempts = []AttemptStatus{}
	}
	if out.ActiveCampaigns == nil {
		out.ActiveCampaigns = []CampaignStatus{}
	}
	if out.CurrentModels == nil {
		out.CurrentModels = []CurrentModel{}
	}
	if out.CurrentProducts == nil {
		out.CurrentProducts = []CurrentProduct{}
	}
	return out, nil
}
