package experiment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type State string

const (
	StateDraft         State = "DRAFT"
	StatePreregistered State = "PREREGISTERED"
	StateRunning       State = "RUNNING"
	StateCompleted     State = "COMPLETED"
	StateSuperseded    State = "SUPERSEDED"
	StateAborted       State = "ABORTED"
)

type CreateRequest struct {
	Slug                   string         `json:"slug"`
	Name                   string         `json:"name"`
	ResearchQuestion       string         `json:"research_question"`
	Hypothesis             string         `json:"hypothesis"`
	InitialState           State          `json:"initial_state"`
	SupersedesExperimentID *uuid.UUID     `json:"supersedes_experiment_id,omitempty"`
	Metadata               map[string]any `json:"metadata,omitempty"`
	CreatedBy              string         `json:"created_by"`
}

type TransitionRequest struct {
	ExperimentID uuid.UUID `json:"experiment_id"`
	ToState      State     `json:"to_state"`
	Actor        string    `json:"actor"`
}

type Record struct {
	ID                     uuid.UUID      `json:"id"`
	Slug                   string         `json:"slug"`
	Name                   string         `json:"name"`
	ResearchQuestion       string         `json:"research_question"`
	Hypothesis             string         `json:"hypothesis"`
	LifecycleState         State          `json:"lifecycle_state"`
	SpecSHA256             string         `json:"spec_sha256"`
	SupersedesExperimentID *uuid.UUID     `json:"supersedes_experiment_id,omitempty"`
	Metadata               map[string]any `json:"metadata"`
	CreatedBy              string         `json:"created_by"`
	CreatedAt              time.Time      `json:"created_at"`
	PreregisteredAt        *time.Time     `json:"preregistered_at,omitempty"`
	CompletedAt            *time.Time     `json:"completed_at,omitempty"`
}

type Created struct {
	Experiment Record `json:"experiment"`
	Reused     bool   `json:"reused"`
}

var ErrNotFound = errors.New("experiment not found")

func normalizeCreate(req CreateRequest) (CreateRequest, error) {
	req.Slug = strings.TrimSpace(req.Slug)
	req.Name = strings.TrimSpace(req.Name)
	req.ResearchQuestion = strings.TrimSpace(req.ResearchQuestion)
	req.Hypothesis = strings.TrimSpace(req.Hypothesis)
	req.CreatedBy = strings.TrimSpace(req.CreatedBy)
	if req.Metadata == nil {
		req.Metadata = map[string]any{}
	}
	if req.InitialState == "" {
		req.InitialState = StateDraft
	}
	if req.Slug == "" || req.Name == "" || req.ResearchQuestion == "" || req.Hypothesis == "" || req.CreatedBy == "" {
		return CreateRequest{}, errors.New("experiment slug/name/question/hypothesis/created_by are required")
	}
	if req.InitialState != StateDraft && req.InitialState != StatePreregistered {
		return CreateRequest{}, errors.New("experiment initial_state must be DRAFT or PREREGISTERED")
	}
	return req, nil
}

func specSHA(req CreateRequest) (string, error) {
	spec := map[string]any{
		"schema":            "RealSaS.ExperimentSpec.v1",
		"slug":              req.Slug,
		"name":              req.Name,
		"research_question": req.ResearchQuestion,
		"hypothesis":        req.Hypothesis,
		"metadata":          req.Metadata,
	}
	if req.SupersedesExperimentID != nil {
		spec["supersedes_experiment_id"] = req.SupersedesExperimentID.String()
	}
	return semantic.JSONSHA256(spec)
}

func Create(ctx context.Context, pool *pgxpool.Pool, raw CreateRequest) (Created, error) {
	req, err := normalizeCreate(raw)
	if err != nil {
		return Created{}, err
	}
	sha, err := specSHA(req)
	if err != nil {
		return Created{}, err
	}
	metadata, err := json.Marshal(req.Metadata)
	if err != nil {
		return Created{}, err
	}
	id := uuid.New()
	var reused bool
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var existingID uuid.UUID
		var existingSHA string
		err := tx.QueryRow(ctx, "SELECT id,spec_sha256 FROM experiments WHERE slug=$1 FOR UPDATE", req.Slug).Scan(&existingID, &existingSHA)
		switch {
		case err == nil:
			if existingSHA != sha {
				return errors.New("EXPERIMENT_SLUG_SPEC_DRIFT")
			}
			id = existingID
			reused = true
			return nil
		case !errors.Is(err, pgx.ErrNoRows):
			return err
		}
		if req.SupersedesExperimentID != nil {
			var prior uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT id FROM experiments WHERE id=$1 FOR SHARE", *req.SupersedesExperimentID).Scan(&prior); err != nil {
				return fmt.Errorf("superseded experiment: %w", err)
			}
		}
		var preregisteredAt any
		if req.InitialState == StatePreregistered {
			preregisteredAt = time.Now().UTC()
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO experiments
			  (id,slug,name,research_question,hypothesis,lifecycle_state,spec_sha256,
			   supersedes_experiment_id,metadata,created_by,preregistered_at)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
		`, id, req.Slug, req.Name, req.ResearchQuestion, req.Hypothesis, string(req.InitialState), sha,
			req.SupersedesExperimentID, metadata, req.CreatedBy, preregisteredAt); err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{"experiment_id": id.String(), "slug": req.Slug, "state": req.InitialState, "spec_sha256": sha})
		_, err = tx.Exec(ctx, "INSERT INTO audit_events(actor,action,payload) VALUES ($1,'EXPERIMENT_CREATED',$2)", req.CreatedBy, audit)
		return err
	})
	if err != nil {
		return Created{}, err
	}
	record, err := Get(ctx, pool, id)
	if err != nil {
		return Created{}, err
	}
	return Created{Experiment: record, Reused: reused}, nil
}

func allowedTransition(from, to State) bool {
	switch from {
	case StateDraft:
		return to == StatePreregistered || to == StateAborted
	case StatePreregistered:
		return to == StateRunning || to == StateAborted || to == StateSuperseded
	case StateRunning:
		return to == StateCompleted || to == StateAborted || to == StateSuperseded
	case StateCompleted:
		return to == StateSuperseded
	default:
		return false
	}
}

func Transition(ctx context.Context, pool *pgxpool.Pool, req TransitionRequest) (Record, error) {
	if req.ExperimentID == uuid.Nil || strings.TrimSpace(req.Actor) == "" {
		return Record{}, errors.New("experiment_id and actor are required")
	}
	var from State
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var raw string
		if err := tx.QueryRow(ctx, "SELECT lifecycle_state FROM experiments WHERE id=$1 FOR UPDATE", req.ExperimentID).Scan(&raw); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrNotFound
			}
			return err
		}
		from = State(raw)
		if from == req.ToState {
			return nil
		}
		if !allowedTransition(from, req.ToState) {
			return fmt.Errorf("EXPERIMENT_STATE_TRANSITION_FORBIDDEN:%s->%s", from, req.ToState)
		}
		_, err := tx.Exec(ctx, `
			UPDATE experiments SET lifecycle_state=$2,
			preregistered_at=CASE WHEN $2='PREREGISTERED' AND preregistered_at IS NULL THEN now() ELSE preregistered_at END,
			completed_at=CASE WHEN $2 IN ('COMPLETED','ABORTED','SUPERSEDED') THEN now() ELSE completed_at END
			WHERE id=$1
		`, req.ExperimentID, string(req.ToState))
		if err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{"experiment_id": req.ExperimentID.String(), "from": from, "to": req.ToState})
		_, err = tx.Exec(ctx, "INSERT INTO audit_events(actor,action,payload) VALUES ($1,'EXPERIMENT_STATE_CHANGED',$2)", req.Actor, audit)
		return err
	})
	if err != nil {
		return Record{}, err
	}
	return Get(ctx, pool, req.ExperimentID)
}

func scanRecord(row pgx.Row) (Record, error) {
	var out Record
	var state string
	var metadata []byte
	if err := row.Scan(&out.ID, &out.Slug, &out.Name, &out.ResearchQuestion, &out.Hypothesis, &state, &out.SpecSHA256,
		&out.SupersedesExperimentID, &metadata, &out.CreatedBy, &out.CreatedAt, &out.PreregisteredAt, &out.CompletedAt); err != nil {
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

func Get(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (Record, error) {
	return scanRecord(pool.QueryRow(ctx, `
		SELECT id,slug,name,research_question,hypothesis,lifecycle_state,spec_sha256,
		       supersedes_experiment_id,metadata,created_by,created_at,preregistered_at,completed_at
		FROM experiments WHERE id=$1
	`, id))
}

func List(ctx context.Context, pool *pgxpool.Pool) ([]Record, error) {
	rows, err := pool.Query(ctx, `
		SELECT id,slug,name,research_question,hypothesis,lifecycle_state,spec_sha256,
		       supersedes_experiment_id,metadata,created_by,created_at,preregistered_at,completed_at
		FROM experiments ORDER BY created_at DESC,id
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
		out = append(out, record)
	}
	return out, rows.Err()
}
