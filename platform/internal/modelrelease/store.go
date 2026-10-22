package modelrelease

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type State string
type Channel string

const (
	StateDraft             State = "DRAFT"
	StateResearchQualified State = "RESEARCH_QUALIFIED"
	StateProductCandidate  State = "PRODUCT_CANDIDATE"
	StateProduction        State = "PRODUCTION"
	StateSuperseded        State = "SUPERSEDED"
	StateRejected          State = "REJECTED"

	ChannelResearch   Channel = "RESEARCH"
	ChannelCandidate  Channel = "CANDIDATE"
	ChannelProduction Channel = "PRODUCTION"
)

type RegisterRequest struct {
	Family                   string         `json:"family"`
	Name                     string         `json:"name"`
	ArchitectureID           string         `json:"architecture_id"`
	CheckpointArtifactID     uuid.UUID      `json:"checkpoint_artifact_id"`
	EngineReleaseID          uuid.UUID      `json:"engine_release_id"`
	OriginatingExperimentID  *uuid.UUID     `json:"originating_experiment_id,omitempty"`
	OriginatingAttemptID     *uuid.UUID     `json:"originating_attempt_id,omitempty"`
	SourceCodeSHA            string         `json:"source_code_sha"`
	InitialState             State          `json:"initial_state"`
	SupersedesModelReleaseID *uuid.UUID     `json:"supersedes_model_release_id,omitempty"`
	Metadata                 map[string]any `json:"metadata,omitempty"`
	CreatedBy                string         `json:"created_by"`
}

type EvaluationRequest struct {
	ModelReleaseID     uuid.UUID      `json:"model_release_id"`
	Scope              string         `json:"scope"`
	Result             string         `json:"result"`
	EvidenceArtifactID *uuid.UUID     `json:"evidence_artifact_id,omitempty"`
	ExperimentID       *uuid.UUID     `json:"experiment_id,omitempty"`
	Details            map[string]any `json:"details,omitempty"`
	Actor              string         `json:"actor"`
}

type PromoteRequest struct {
	Family                   string    `json:"family"`
	Channel                  Channel   `json:"channel"`
	ModelReleaseID           uuid.UUID `json:"model_release_id"`
	RequiredEvaluationScopes []string  `json:"required_evaluation_scopes,omitempty"`
	ExpectedLockVersion      *int64    `json:"expected_lock_version,omitempty"`
	Actor                    string    `json:"actor"`
}

type Evaluation struct {
	Scope              string         `json:"scope"`
	Result             string         `json:"result"`
	EvidenceArtifactID *uuid.UUID     `json:"evidence_artifact_id,omitempty"`
	ExperimentID       *uuid.UUID     `json:"experiment_id,omitempty"`
	Details            map[string]any `json:"details"`
	ObservedAt         *time.Time     `json:"observed_at,omitempty"`
	UpdatedAt          time.Time      `json:"updated_at"`
}

type Record struct {
	ID                       uuid.UUID      `json:"id"`
	Family                   string         `json:"family"`
	Name                     string         `json:"name"`
	ArchitectureID           string         `json:"architecture_id"`
	CheckpointArtifactID     uuid.UUID      `json:"checkpoint_artifact_id"`
	EngineReleaseID          uuid.UUID      `json:"engine_release_id"`
	OriginatingExperimentID  *uuid.UUID     `json:"originating_experiment_id,omitempty"`
	OriginatingAttemptID     *uuid.UUID     `json:"originating_attempt_id,omitempty"`
	SourceCodeSHA            string         `json:"source_code_sha"`
	LifecycleState           State          `json:"lifecycle_state"`
	SupersedesModelReleaseID *uuid.UUID     `json:"supersedes_model_release_id,omitempty"`
	Metadata                 map[string]any `json:"metadata"`
	CreatedBy                string         `json:"created_by"`
	CreatedAt                time.Time      `json:"created_at"`
	SealedAt                 time.Time      `json:"sealed_at"`
	Evaluations              []Evaluation   `json:"evaluations,omitempty"`
}

type ChannelRecord struct {
	Family         string    `json:"family"`
	Channel        Channel   `json:"channel"`
	ModelReleaseID uuid.UUID `json:"model_release_id"`
	LockVersion    int64     `json:"lock_version"`
	UpdatedAt      time.Time `json:"updated_at"`
}

var ErrNotFound = errors.New("model release not found")

func validateSHA40(value string) error {
	if len(value) != 40 {
		return errors.New("source_code_sha must be 40 lowercase hex characters")
	}
	if strings.ToLower(value) != value {
		return errors.New("source_code_sha must be lowercase hex")
	}
	_, err := hex.DecodeString(value)
	return err
}

func normalizeRegister(req RegisterRequest) (RegisterRequest, error) {
	req.Family = strings.TrimSpace(req.Family)
	req.Name = strings.TrimSpace(req.Name)
	req.ArchitectureID = strings.TrimSpace(req.ArchitectureID)
	req.SourceCodeSHA = strings.TrimSpace(req.SourceCodeSHA)
	req.CreatedBy = strings.TrimSpace(req.CreatedBy)
	if req.Metadata == nil {
		req.Metadata = map[string]any{}
	}
	if req.InitialState == "" {
		req.InitialState = StateDraft
	}
	if req.Family == "" || req.Name == "" || req.ArchitectureID == "" || req.CreatedBy == "" {
		return RegisterRequest{}, errors.New("family/name/architecture_id/created_by are required")
	}
	if req.CheckpointArtifactID == uuid.Nil || req.EngineReleaseID == uuid.Nil {
		return RegisterRequest{}, errors.New("checkpoint_artifact_id and engine_release_id are required")
	}
	if err := validateSHA40(req.SourceCodeSHA); err != nil {
		return RegisterRequest{}, err
	}
	if req.InitialState != StateDraft && req.InitialState != StateResearchQualified {
		return RegisterRequest{}, errors.New("model release initial_state must be DRAFT or RESEARCH_QUALIFIED")
	}
	return req, nil
}

func Register(ctx context.Context, pool *pgxpool.Pool, raw RegisterRequest) (Record, error) {
	req, err := normalizeRegister(raw)
	if err != nil {
		return Record{}, err
	}
	metadata, err := json.Marshal(req.Metadata)
	if err != nil {
		return Record{}, err
	}
	id := uuid.New()
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var artifact uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM artifacts WHERE id=$1 FOR SHARE", req.CheckpointArtifactID).Scan(&artifact); err != nil {
			return fmt.Errorf("checkpoint artifact: %w", err)
		}
		var release uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM engine_releases WHERE id=$1 FOR SHARE", req.EngineReleaseID).Scan(&release); err != nil {
			return fmt.Errorf("engine release: %w", err)
		}
		if req.OriginatingExperimentID != nil {
			var experiment uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT id FROM experiments WHERE id=$1 FOR SHARE", *req.OriginatingExperimentID).Scan(&experiment); err != nil {
				return fmt.Errorf("originating experiment: %w", err)
			}
		}
		if req.OriginatingAttemptID != nil {
			var attemptRelease uuid.UUID
			var attemptExperiment *uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT engine_release_id,experiment_id FROM attempts WHERE id=$1 FOR SHARE", *req.OriginatingAttemptID).Scan(&attemptRelease, &attemptExperiment); err != nil {
				return fmt.Errorf("originating attempt: %w", err)
			}
			if attemptRelease != req.EngineReleaseID {
				return errors.New("MODEL_RELEASE_ATTEMPT_ENGINE_RELEASE_MISMATCH")
			}
			if req.OriginatingExperimentID != nil && (attemptExperiment == nil || *attemptExperiment != *req.OriginatingExperimentID) {
				return errors.New("MODEL_RELEASE_ATTEMPT_EXPERIMENT_MISMATCH")
			}
		}
		if req.SupersedesModelReleaseID != nil {
			var priorFamily string
			if err := tx.QueryRow(ctx, "SELECT family FROM model_releases WHERE id=$1 FOR SHARE", *req.SupersedesModelReleaseID).Scan(&priorFamily); err != nil {
				return fmt.Errorf("superseded model release: %w", err)
			}
			if priorFamily != req.Family {
				return errors.New("MODEL_RELEASE_SUPERSEDES_FAMILY_MISMATCH")
			}
		}
		var existing uuid.UUID
		err := tx.QueryRow(ctx, "SELECT id FROM model_releases WHERE family=$1 AND name=$2", req.Family, req.Name).Scan(&existing)
		if err == nil {
			return errors.New("MODEL_RELEASE_NAME_ALREADY_EXISTS")
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO model_releases
			  (id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
			   originating_experiment_id,originating_attempt_id,source_code_sha,lifecycle_state,
			   supersedes_model_release_id,metadata,created_by)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
		`, id, req.Family, req.Name, req.ArchitectureID, req.CheckpointArtifactID, req.EngineReleaseID,
			req.OriginatingExperimentID, req.OriginatingAttemptID, req.SourceCodeSHA, string(req.InitialState),
			req.SupersedesModelReleaseID, metadata, req.CreatedBy); err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{"model_release_id": id.String(), "family": req.Family, "name": req.Name, "checkpoint_artifact_id": req.CheckpointArtifactID.String()})
		_, err = tx.Exec(ctx, "INSERT INTO audit_events(actor,action,artifact_id,payload) VALUES ($1,'MODEL_RELEASE_REGISTERED',$2,$3)", req.CreatedBy, req.CheckpointArtifactID, audit)
		return err
	})
	if err != nil {
		return Record{}, err
	}
	return Get(ctx, pool, id)
}

func SetEvaluation(ctx context.Context, pool *pgxpool.Pool, req EvaluationRequest) (Record, error) {
	req.Scope = strings.TrimSpace(req.Scope)
	req.Result = strings.TrimSpace(req.Result)
	req.Actor = strings.TrimSpace(req.Actor)
	if req.ModelReleaseID == uuid.Nil || req.Scope == "" || req.Actor == "" {
		return Record{}, errors.New("model_release_id/scope/actor are required")
	}
	if req.Details == nil {
		req.Details = map[string]any{}
	}
	switch req.Result {
	case "NOT_RUN":
		if req.EvidenceArtifactID != nil || req.ExperimentID != nil {
			return Record{}, errors.New("NOT_RUN evaluation cannot carry evidence")
		}
	case "PASS", "FAIL", "ABSTAIN":
		if req.EvidenceArtifactID == nil && req.ExperimentID == nil {
			return Record{}, errors.New("observed evaluation requires evidence_artifact_id or experiment_id")
		}
	default:
		return Record{}, errors.New("evaluation result must be NOT_RUN/PASS/FAIL/ABSTAIN")
	}
	details, err := json.Marshal(req.Details)
	if err != nil {
		return Record{}, err
	}
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var found uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM model_releases WHERE id=$1 FOR UPDATE", req.ModelReleaseID).Scan(&found); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrNotFound
			}
			return err
		}
		if req.EvidenceArtifactID != nil {
			var artifact uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT id FROM artifacts WHERE id=$1 FOR SHARE", *req.EvidenceArtifactID).Scan(&artifact); err != nil {
				return err
			}
		}
		if req.ExperimentID != nil {
			var experiment uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT id FROM experiments WHERE id=$1 FOR SHARE", *req.ExperimentID).Scan(&experiment); err != nil {
				return err
			}
		}
		var observed any
		if req.Result != "NOT_RUN" {
			observed = time.Now().UTC()
		}
		_, err := tx.Exec(ctx, `
			INSERT INTO model_release_evaluations
			  (model_release_id,scope,result,evidence_artifact_id,experiment_id,details,observed_at,updated_at)
			VALUES ($1,$2,$3,$4,$5,$6,$7,now())
			ON CONFLICT (model_release_id,scope) DO UPDATE SET
			  result=EXCLUDED.result,evidence_artifact_id=EXCLUDED.evidence_artifact_id,
			  experiment_id=EXCLUDED.experiment_id,details=EXCLUDED.details,
			  observed_at=EXCLUDED.observed_at,updated_at=now()
		`, req.ModelReleaseID, req.Scope, req.Result, req.EvidenceArtifactID, req.ExperimentID, details, observed)
		if err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{"model_release_id": req.ModelReleaseID.String(), "scope": req.Scope, "result": req.Result})
		_, err = tx.Exec(ctx, "INSERT INTO audit_events(actor,action,artifact_id,payload) VALUES ($1,'MODEL_RELEASE_EVALUATION_SET',$2,$3)", req.Actor, req.EvidenceArtifactID, audit)
		return err
	})
	if err != nil {
		return Record{}, err
	}
	return Get(ctx, pool, req.ModelReleaseID)
}

func targetState(channel Channel) (State, error) {
	switch channel {
	case ChannelResearch:
		return StateResearchQualified, nil
	case ChannelCandidate:
		return StateProductCandidate, nil
	case ChannelProduction:
		return StateProduction, nil
	default:
		return "", errors.New("channel must be RESEARCH/CANDIDATE/PRODUCTION")
	}
}

func Promote(ctx context.Context, pool *pgxpool.Pool, req PromoteRequest) (ChannelRecord, error) {
	req.Family = strings.TrimSpace(req.Family)
	req.Actor = strings.TrimSpace(req.Actor)
	if req.Family == "" || req.ModelReleaseID == uuid.Nil || req.Actor == "" {
		return ChannelRecord{}, errors.New("family/model_release_id/actor are required")
	}
	state, err := targetState(req.Channel)
	if err != nil {
		return ChannelRecord{}, err
	}
	if req.Channel != ChannelResearch && len(req.RequiredEvaluationScopes) == 0 {
		return ChannelRecord{}, errors.New("candidate/production promotion requires explicit evaluation scopes")
	}
	var out ChannelRecord
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var family string
		var currentState string
		if err := tx.QueryRow(ctx, "SELECT family,lifecycle_state FROM model_releases WHERE id=$1 FOR UPDATE", req.ModelReleaseID).Scan(&family, &currentState); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrNotFound
			}
			return err
		}
		if family != req.Family {
			return errors.New("MODEL_CHANNEL_FAMILY_MISMATCH")
		}
		if State(currentState) == StateRejected || State(currentState) == StateSuperseded {
			return errors.New("MODEL_RELEASE_NOT_PROMOTABLE")
		}
		for _, scope := range req.RequiredEvaluationScopes {
			scope = strings.TrimSpace(scope)
			if scope == "" {
				return errors.New("empty required evaluation scope")
			}
			var result string
			if err := tx.QueryRow(ctx, "SELECT result FROM model_release_evaluations WHERE model_release_id=$1 AND scope=$2", req.ModelReleaseID, scope).Scan(&result); err != nil {
				return fmt.Errorf("MODEL_PROMOTION_EVALUATION_MISSING:%s", scope)
			}
			if result != "PASS" {
				return fmt.Errorf("MODEL_PROMOTION_EVALUATION_NOT_PASS:%s:%s", scope, result)
			}
		}
		var existingID uuid.UUID
		var lockVersion int64
		err := tx.QueryRow(ctx, "SELECT model_release_id,lock_version FROM model_channels WHERE family=$1 AND channel=$2 FOR UPDATE", req.Family, string(req.Channel)).Scan(&existingID, &lockVersion)
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		if req.ExpectedLockVersion != nil {
			if errors.Is(err, pgx.ErrNoRows) && *req.ExpectedLockVersion != 0 {
				return errors.New("MODEL_CHANNEL_LOCK_MISMATCH")
			}
			if err == nil && lockVersion != *req.ExpectedLockVersion {
				return errors.New("MODEL_CHANNEL_LOCK_MISMATCH")
			}
		}
		if errors.Is(err, pgx.ErrNoRows) {
			lockVersion = 0
			if _, err := tx.Exec(ctx, "INSERT INTO model_channels(family,channel,model_release_id,lock_version) VALUES ($1,$2,$3,0)", req.Family, string(req.Channel), req.ModelReleaseID); err != nil {
				return err
			}
		} else {
			lockVersion++
			if _, err := tx.Exec(ctx, "UPDATE model_channels SET model_release_id=$3,lock_version=$4,updated_at=now() WHERE family=$1 AND channel=$2", req.Family, string(req.Channel), req.ModelReleaseID, lockVersion); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, "UPDATE model_releases SET lifecycle_state=$2 WHERE id=$1", req.ModelReleaseID, string(state)); err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{"family": req.Family, "channel": req.Channel, "model_release_id": req.ModelReleaseID.String(), "required_evaluation_scopes": req.RequiredEvaluationScopes})
		if _, err := tx.Exec(ctx, "INSERT INTO audit_events(actor,action,payload) VALUES ($1,'MODEL_CHANNEL_PROMOTED',$2)", req.Actor, audit); err != nil {
			return err
		}
		out = ChannelRecord{Family: req.Family, Channel: req.Channel, ModelReleaseID: req.ModelReleaseID, LockVersion: lockVersion}
		return tx.QueryRow(ctx, "SELECT updated_at FROM model_channels WHERE family=$1 AND channel=$2", req.Family, string(req.Channel)).Scan(&out.UpdatedAt)
	})
	return out, err
}

func scanRecord(row pgx.Row) (Record, error) {
	var out Record
	var state string
	var metadata []byte
	if err := row.Scan(&out.ID, &out.Family, &out.Name, &out.ArchitectureID, &out.CheckpointArtifactID,
		&out.EngineReleaseID, &out.OriginatingExperimentID, &out.OriginatingAttemptID, &out.SourceCodeSHA,
		&state, &out.SupersedesModelReleaseID, &metadata, &out.CreatedBy, &out.CreatedAt, &out.SealedAt); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Record{}, ErrNotFound
		}
		return Record{}, err
	}
	out.LifecycleState = State(state)
	if err := json.Unmarshal(metadata, &out.Metadata); err != nil {
		return Record{}, err
	}
	return out, nil
}

func loadEvaluations(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) ([]Evaluation, error) {
	rows, err := pool.Query(ctx, `
		SELECT scope,result,evidence_artifact_id,experiment_id,details,observed_at,updated_at
		FROM model_release_evaluations WHERE model_release_id=$1 ORDER BY scope
	`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Evaluation, 0)
	for rows.Next() {
		var evaluation Evaluation
		var details []byte
		if err := rows.Scan(&evaluation.Scope, &evaluation.Result, &evaluation.EvidenceArtifactID, &evaluation.ExperimentID, &details, &evaluation.ObservedAt, &evaluation.UpdatedAt); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(details, &evaluation.Details); err != nil {
			return nil, err
		}
		out = append(out, evaluation)
	}
	return out, rows.Err()
}

func Get(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (Record, error) {
	out, err := scanRecord(pool.QueryRow(ctx, `
		SELECT id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
		       originating_experiment_id,originating_attempt_id,source_code_sha,lifecycle_state,
		       supersedes_model_release_id,metadata,created_by,created_at,sealed_at
		FROM model_releases WHERE id=$1
	`, id))
	if err != nil {
		return Record{}, err
	}
	out.Evaluations, err = loadEvaluations(ctx, pool, id)
	return out, err
}

func List(ctx context.Context, pool *pgxpool.Pool) ([]Record, error) {
	rows, err := pool.Query(ctx, `
		SELECT id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
		       originating_experiment_id,originating_attempt_id,source_code_sha,lifecycle_state,
		       supersedes_model_release_id,metadata,created_by,created_at,sealed_at
		FROM model_releases ORDER BY family,created_at DESC,id
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Record, 0)
	for rows.Next() {
		record, err := scanRecord(rows)
		if err != nil {
			return nil, err
		}
		record.Evaluations, err = loadEvaluations(ctx, pool, record.ID)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

func Channels(ctx context.Context, pool *pgxpool.Pool) ([]ChannelRecord, error) {
	rows, err := pool.Query(ctx, "SELECT family,channel,model_release_id,lock_version,updated_at FROM model_channels ORDER BY family,channel")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]ChannelRecord, 0)
	for rows.Next() {
		var record ChannelRecord
		var channel string
		if err := rows.Scan(&record.Family, &channel, &record.ModelReleaseID, &record.LockVersion, &record.UpdatedAt); err != nil {
			return nil, err
		}
		record.Channel = Channel(channel)
		out = append(out, record)
	}
	return out, rows.Err()
}
