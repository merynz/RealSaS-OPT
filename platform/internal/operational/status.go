package operational

import (
	"context"
	"fmt"
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

type CurrentProduct struct {
	SubjectID           uuid.UUID             `json:"subject_id"`
	SubjectSlug         string                `json:"subject_slug"`
	ProductRevisionID   uuid.UUID             `json:"product_revision_id"`
	RevisionNumber      int64                 `json:"revision_number"`
	ManifestSHA256      string                `json:"manifest_sha256"`
	EngineReleaseID     *uuid.UUID            `json:"engine_release_id,omitempty"`
	EngineReleaseSHA256 *string               `json:"engine_release_sha256,omitempty"`
	CapabilitySetSHA256 *string               `json:"capability_set_sha256,omitempty"`
	Models              []ProductModelBinding `json:"models"`
	UpdatedAt           time.Time             `json:"updated_at"`
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

	productRows, err := pool.Query(ctx, `
		SELECT c.subject_id,s.slug,c.product_revision_id,r.revision_number,r.manifest_sha256,
		       r.engine_release_id,er.release_sha256,r.capability_set_sha256,c.updated_at
		FROM subject_current_revision c
		JOIN subjects s ON s.id=c.subject_id
		JOIN product_revisions r ON r.id=c.product_revision_id
		LEFT JOIN engine_releases er ON er.id=r.engine_release_id
		ORDER BY s.slug,c.subject_id
	`)
	if err != nil {
		return Snapshot{}, err
	}
	productIndex := map[uuid.UUID]int{}
	for productRows.Next() {
		var record CurrentProduct
		if err := productRows.Scan(
			&record.SubjectID,
			&record.SubjectSlug,
			&record.ProductRevisionID,
			&record.RevisionNumber,
			&record.ManifestSHA256,
			&record.EngineReleaseID,
			&record.EngineReleaseSHA256,
			&record.CapabilitySetSHA256,
			&record.UpdatedAt,
		); err != nil {
			productRows.Close()
			return Snapshot{}, err
		}
		record.Models = []ProductModelBinding{}
		productIndex[record.ProductRevisionID] = len(out.CurrentProducts)
		out.CurrentProducts = append(out.CurrentProducts, record)
	}
	productRows.Close()
	if err := productRows.Err(); err != nil {
		return Snapshot{}, err
	}

	modelRows, err := pool.Query(ctx, `
		SELECT prm.product_revision_id,prm.family,prm.model_release_id,
		       mr.architecture_id,mr.checkpoint_artifact_id,a.content_sha256,mr.lifecycle_state
		FROM product_revision_model_releases prm
		JOIN subject_current_revision c ON c.product_revision_id=prm.product_revision_id
		JOIN model_releases mr ON mr.id=prm.model_release_id
		JOIN artifacts a ON a.id=mr.checkpoint_artifact_id
		ORDER BY prm.product_revision_id,prm.family
	`)
	if err != nil {
		return Snapshot{}, err
	}
	for modelRows.Next() {
		var revisionID uuid.UUID
		var binding ProductModelBinding
		if err := modelRows.Scan(
			&revisionID,
			&binding.Family,
			&binding.ModelReleaseID,
			&binding.ArchitectureID,
			&binding.CheckpointArtifactID,
			&binding.CheckpointContentSHA256,
			&binding.LifecycleState,
		); err != nil {
			modelRows.Close()
			return Snapshot{}, err
		}
		index, ok := productIndex[revisionID]
		if !ok {
			modelRows.Close()
			return Snapshot{}, fmt.Errorf("CURRENT_PRODUCT_MODEL_BINDING_WITHOUT_CURRENT_REVISION:%s", revisionID)
		}
		out.CurrentProducts[index].Models = append(out.CurrentProducts[index].Models, binding)
	}
	modelRows.Close()
	if err := modelRows.Err(); err != nil {
		return Snapshot{}, err
	}

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
