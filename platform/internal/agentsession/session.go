// Package agentsession binds agent handoffs to durable Attempts, never branches.
package agentsession

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type Scope struct {
	Schema             string    `json:"schema"`
	SessionID          uuid.UUID `json:"session_id"`
	AttemptID          uuid.UUID `json:"attempt_id"`
	SubjectID          uuid.UUID `json:"subject_id"`
	EngineReleaseID    uuid.UUID `json:"engine_release_id"`
	ReleaseSHA256      string    `json:"release_sha256"`
	AttemptSpecSHA256  string    `json:"attempt_spec_sha256"`
	SubjectInputID     uuid.UUID `json:"subject_input_id"`
	SubjectInputSHA256 string    `json:"subject_input_sha256"`
	TargetStageID      string    `json:"target_stage_id"`
	CanonicalLine      string    `json:"canonical_line"`
	CanonicalCodeSHA   string    `json:"canonical_code_sha"`
	OpenedBy           string    `json:"opened_by"`
}

type OpenRequest struct {
	AttemptID             uuid.UUID  `json:"attempt_id"`
	SubjectInputID        uuid.UUID  `json:"subject_input_id"`
	TargetStageID         string     `json:"target_stage_id"`
	CanonicalCodeSHA      string     `json:"canonical_code_sha"`
	ExpectedHandoffSHA256 string     `json:"expected_handoff_sha256"`
	ResumeSessionID       *uuid.UUID `json:"resume_session_id,omitempty"`
	CreatedBy             string     `json:"created_by"`
}

type Ticket struct {
	Scope       Scope  `json:"scope"`
	ScopeSHA256 string `json:"scope_sha256"`
	Resumed     bool   `json:"resumed"`
}

type CloseRequest struct {
	SessionID   uuid.UUID `json:"session_id"`
	AttemptID   uuid.UUID `json:"attempt_id"`
	ScopeSHA256 string    `json:"scope_sha256"`
	NextAction  string    `json:"next_action"`
	Summary     string    `json:"summary"`
	CreatedBy   string    `json:"created_by"`
}

func ValidateCode(actual, expected string) error {
	if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(actual) || actual != expected {
		return errors.New("AGENT_CANONICAL_CODE_MISMATCH__DEPLOY_CURRENT_MAIN")
	}
	return nil
}

func lockSubject(ctx context.Context, tx pgx.Tx, subject uuid.UUID) error {
	_, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "agent-session:"+subject.String())
	return err
}

func active(ctx context.Context, tx pgx.Tx, subject uuid.UUID) (*Scope, error) {
	var raw []byte
	err := tx.QueryRow(ctx, `SELECT e.payload FROM attempt_events e JOIN attempts a ON a.id=e.attempt_id
		WHERE a.subject_id=$1 AND e.event_type='AGENT_SESSION_OPENED' AND NOT EXISTS (
		SELECT 1 FROM attempt_events c WHERE c.attempt_id=e.attempt_id AND c.event_type='AGENT_HANDOFF_SEALED'
		AND c.payload->'scope'->>'session_id'=e.payload->>'session_id') ORDER BY e.id DESC LIMIT 1`, subject).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var scope Scope
	err = json.Unmarshal(raw, &scope)
	return &scope, err
}

func lastHandoff(ctx context.Context, tx pgx.Tx, subject uuid.UUID) (map[string]any, error) {
	var raw []byte
	err := tx.QueryRow(ctx, `SELECT e.payload FROM attempt_events e JOIN attempts a ON a.id=e.attempt_id
		WHERE a.subject_id=$1 AND e.event_type='AGENT_HANDOFF_SEALED' ORDER BY e.id DESC LIMIT 1`, subject).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var value map[string]any
	err = json.Unmarshal(raw, &value)
	return value, err
}

func Open(ctx context.Context, pool *pgxpool.Pool, deployedCode string, req OpenRequest) (Ticket, error) {
	if err := ValidateCode(deployedCode, req.CanonicalCodeSHA); err != nil {
		return Ticket{}, err
	}
	if req.AttemptID == uuid.Nil || req.SubjectInputID == uuid.Nil || req.TargetStageID == "" || req.CreatedBy == "" {
		return Ticket{}, errors.New("AGENT_ENTRY_SCOPE_REQUIRED")
	}
	var scope Scope
	var parent *uuid.UUID
	err := pool.QueryRow(ctx, `SELECT a.subject_id,a.engine_release_id,a.spec_sha256,a.parent_attempt_id,r.release_sha256
		FROM attempts a JOIN engine_releases r ON r.id=a.engine_release_id WHERE a.id=$1`, req.AttemptID).
		Scan(&scope.SubjectID, &scope.EngineReleaseID, &scope.AttemptSpecSHA256, &parent, &scope.ReleaseSHA256)
	if err != nil {
		return Ticket{}, err
	}
	sealedInput, err := input.Load(ctx, pool, req.SubjectInputID, scope.SubjectID)
	if err != nil {
		return Ticket{}, err
	}
	scope.SubjectInputSHA256 = sealedInput.ManifestSHA256
	graph, _, err := release.LoadGraph(ctx, pool, scope.EngineReleaseID)
	if err != nil {
		return Ticket{}, err
	}
	if _, ok := graph.Get(req.TargetStageID); !ok {
		return Ticket{}, errors.New("AGENT_TARGET_NOT_IN_RELEASED_DAG")
	}
	scope.Schema = "RealSaS.AgentExecutionScope.v1"
	scope.SessionID, scope.AttemptID, scope.SubjectInputID = uuid.New(), req.AttemptID, req.SubjectInputID
	scope.CanonicalLine, scope.CanonicalCodeSHA, scope.TargetStageID, scope.OpenedBy = "main", deployedCode, req.TargetStageID, req.CreatedBy
	var out Ticket
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if err := lockSubject(ctx, tx, scope.SubjectID); err != nil {
			return err
		}
		existing, err := active(ctx, tx, scope.SubjectID)
		if err != nil {
			return err
		}
		if existing != nil {
			if req.ResumeSessionID == nil || *req.ResumeSessionID != existing.SessionID {
				return fmt.Errorf("AGENT_ACTIVE_SESSION_MUST_RESUME: %s", existing.SessionID)
			}
			if existing.AttemptID != scope.AttemptID || existing.SubjectInputID != scope.SubjectInputID ||
				existing.TargetStageID != scope.TargetStageID || existing.CanonicalCodeSHA != scope.CanonicalCodeSHA {
				return errors.New("AGENT_RESUME_SCOPE_DRIFT")
			}
			hash, err := semantic.JSONSHA256(existing)
			out = Ticket{Scope: *existing, ScopeSHA256: hash, Resumed: true}
			return err
		}
		if req.ResumeSessionID != nil {
			return errors.New("AGENT_RESUME_SESSION_NOT_ACTIVE")
		}
		previous, err := lastHandoff(ctx, tx, scope.SubjectID)
		if err != nil {
			return err
		}
		if previous != nil {
			if previous["handoff_sha256"] != req.ExpectedHandoffSHA256 {
				return errors.New("AGENT_ENTRY_STALE_HANDOFF")
			}
			priorScope, ok := previous["scope"].(map[string]any)
			if !ok {
				return errors.New("AGENT_HANDOFF_SCOPE_INVALID")
			}
			prior, ok := priorScope["attempt_id"].(string)
			if !ok {
				return errors.New("AGENT_HANDOFF_SCOPE_INVALID")
			}
			if prior != scope.AttemptID.String() && (parent == nil || prior != parent.String()) {
				return errors.New("AGENT_ENTRY_ATTEMPT_LINEAGE_DRIFT")
			}
		} else if req.ExpectedHandoffSHA256 != "" {
			return errors.New("AGENT_ENTRY_HANDOFF_NOT_FOUND")
		}
		raw, err := json.Marshal(scope)
		if err != nil {
			return err
		}
		if _, err = tx.Exec(ctx, "INSERT INTO attempt_events(attempt_id,event_type,payload) VALUES ($1,'AGENT_SESSION_OPENED',$2)", scope.AttemptID, raw); err != nil {
			return err
		}
		hash, err := semantic.JSONSHA256(scope)
		out = Ticket{Scope: scope, ScopeSHA256: hash}
		return err
	})
	return out, err
}

func Load(ctx context.Context, tx pgx.Tx, id, attemptID uuid.UUID) (Scope, error) {
	var raw []byte
	err := tx.QueryRow(ctx, "SELECT payload FROM attempt_events WHERE attempt_id=$2 AND event_type='AGENT_SESSION_OPENED' AND payload->>'session_id'=$1 ORDER BY id LIMIT 1", id.String(), attemptID).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return Scope{}, errors.New("AGENT_SESSION_NOT_IN_ATTEMPT")
	}
	if err != nil {
		return Scope{}, err
	}
	var scope Scope
	err = json.Unmarshal(raw, &scope)
	return scope, err
}

func (s Scope) CheckCompile(attemptID, subject, engineRelease, subjectInput uuid.UUID, target string) error {
	if s.CanonicalLine != "main" || s.AttemptID != attemptID || s.SubjectID != subject ||
		s.EngineReleaseID != engineRelease || s.SubjectInputID != subjectInput || s.TargetStageID != target {
		return errors.New("AGENT_COMPILE_SCOPE_DRIFT")
	}
	return nil
}

func CheckCompile(ctx context.Context, tx pgx.Tx, id, attemptID, subject, engineRelease, subjectInput uuid.UUID, target, deployedCode string) error {
	scope, err := Load(ctx, tx, id, attemptID)
	if err != nil {
		return err
	}
	if err := ValidateCode(scope.CanonicalCodeSHA, deployedCode); err != nil {
		return err
	}
	if err := scope.CheckCompile(attemptID, subject, engineRelease, subjectInput, target); err != nil {
		return err
	}
	if err := lockSubject(ctx, tx, subject); err != nil {
		return err
	}
	current, err := active(ctx, tx, subject)
	if err != nil {
		return err
	}
	if current == nil || current.SessionID != id {
		return errors.New("AGENT_SESSION_CLOSED_OR_SUPERSEDED")
	}
	return nil
}

func Close(ctx context.Context, pool *pgxpool.Pool, req CloseRequest) (map[string]any, error) {
	if req.SessionID == uuid.Nil || req.AttemptID == uuid.Nil || strings.TrimSpace(req.NextAction) == "" || req.CreatedBy == "" || len(req.NextAction) > 4096 || len(req.Summary) > 8192 {
		return nil, errors.New("AGENT_EXIT_SCOPE_AND_NEXT_ACTION_REQUIRED")
	}
	var out map[string]any
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		scope, err := Load(ctx, tx, req.SessionID, req.AttemptID)
		if err != nil {
			return err
		}
		if err := lockSubject(ctx, tx, scope.SubjectID); err != nil {
			return err
		}
		hash, err := semantic.JSONSHA256(scope)
		if err != nil {
			return err
		}
		if req.AttemptID != scope.AttemptID || hash != req.ScopeSHA256 {
			return errors.New("AGENT_EXIT_SCOPE_DRIFT")
		}
		var sealed []byte
		err = tx.QueryRow(ctx, `SELECT payload FROM attempt_events WHERE attempt_id=$2 AND event_type='AGENT_HANDOFF_SEALED' AND payload->'scope'->>'session_id'=$1`, req.SessionID.String(), req.AttemptID).Scan(&sealed)
		if err == nil {
			if err := json.Unmarshal(sealed, &out); err != nil {
				return err
			}
			if out["next_action"] != req.NextAction || out["summary"] != req.Summary || out["created_by"] != req.CreatedBy {
				return errors.New("AGENT_EXIT_IDEMPOTENCY_DRIFT")
			}
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		current, err := active(ctx, tx, scope.SubjectID)
		if err != nil {
			return err
		}
		if current == nil || current.SessionID != scope.SessionID {
			return errors.New("AGENT_SESSION_CLOSED_OR_SUPERSEDED")
		}
		var pending int
		if err := tx.QueryRow(ctx, `SELECT count(*) FROM commands c JOIN attempts a ON a.subject_id=c.subject_id AND a.id=$2 WHERE c.subject_id=$3 AND c.payload->>'attempt_id'=$2::text AND c.payload->>'agent_session_id'=$1 AND a.final_state='OPEN'`, req.SessionID.String(), req.AttemptID, scope.SubjectID).Scan(&pending); err != nil {
			return err
		}
		if pending != 0 {
			return errors.New("AGENT_EXIT_COMMAND_STILL_RUNNING__RESUME_AND_INSPECT")
		}
		var state string
		if err := tx.QueryRow(ctx, "SELECT final_state FROM attempts WHERE id=$1", scope.AttemptID).Scan(&state); err != nil {
			return err
		}
		rows, err := tx.Query(ctx, "SELECT role,artifact_id FROM attempt_artifacts WHERE attempt_id=$1 ORDER BY role", scope.AttemptID)
		if err != nil {
			return err
		}
		bindings := map[string]string{}
		for rows.Next() {
			var role string
			var id uuid.UUID
			if err := rows.Scan(&role, &id); err != nil {
				rows.Close()
				return err
			}
			bindings[role] = id.String()
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return err
		}
		out = map[string]any{"schema": "RealSaS.AgentHandoff.v1", "scope": scope, "scope_sha256": hash,
			"attempt_state": state, "artifact_bindings": bindings, "next_action": req.NextAction, "summary": req.Summary, "created_by": req.CreatedBy}
		handoffSHA, err := semantic.JSONSHA256(out)
		if err != nil {
			return err
		}
		out["handoff_sha256"] = handoffSHA
		raw, err := json.Marshal(out)
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, "INSERT INTO attempt_events(attempt_id,event_type,payload) VALUES ($1,'AGENT_HANDOFF_SEALED',$2)", scope.AttemptID, raw)
		return err
	})
	return out, err
}

func Context(ctx context.Context, pool *pgxpool.Pool, subject uuid.UUID, codeSHA string) (map[string]any, error) {
	var out map[string]any
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if err := lockSubject(ctx, tx, subject); err != nil {
			return err
		}
		current, err := active(ctx, tx, subject)
		if err != nil {
			return err
		}
		previous, err := lastHandoff(ctx, tx, subject)
		if err != nil {
			return err
		}
		out = map[string]any{"canonical_line": "main", "canonical_code_sha": codeSHA, "active_session": current, "last_handoff": previous}
		return nil
	})
	return out, err
}
