package externalstage

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type Binding struct {
	Role       string    `json:"role"`
	ArtifactID uuid.UUID `json:"artifact_id"`
}

type BindRequest struct {
	AttemptID uuid.UUID `json:"attempt_id"`
	StageID   string    `json:"stage_id"`
	Bindings  []Binding `json:"bindings"`
	BoundBy   string    `json:"bound_by"`
}

type Bound struct {
	AttemptID        uuid.UUID `json:"attempt_id"`
	StageID          string    `json:"stage_id"`
	EvidenceSHA256   string    `json:"evidence_sha256"`
	WorkflowSignaled bool      `json:"workflow_signaled"`
	Reused           bool      `json:"reused"`
}

type Identity struct {
	Role           string
	ArtifactID     uuid.UUID
	ArtifactType   string
	SemanticSHA256 string
}

func LoadIdentities(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) (map[string][]Identity, error) {
	rows, err := pool.Query(ctx, `
        SELECT e.stage_id,e.role,e.artifact_id,t.name,a.semantic_sha256
        FROM external_stage_evidence e
        JOIN artifacts a ON a.id=e.artifact_id
        JOIN artifact_types t ON t.id=a.artifact_type_id
        WHERE e.attempt_id=$1 ORDER BY e.stage_id,e.role
    `, attemptID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]Identity{}
	for rows.Next() {
		var stageID string
		var x Identity
		if err := rows.Scan(&stageID, &x.Role, &x.ArtifactID, &x.ArtifactType, &x.SemanticSHA256); err != nil {
			return nil, err
		}
		out[stageID] = append(out[stageID], x)
	}
	return out, rows.Err()
}

func declaredExternalRoles(raw []byte) (map[string]struct{}, error) {
	var parameters map[string]any
	if err := json.Unmarshal(raw, &parameters); err != nil {
		return nil, err
	}
	value, ok := parameters["external_evidence_roles"]
	if !ok {
		return nil, errors.New("EXTERNAL_STAGE_ROLE_CONTRACT_REQUIRED")
	}
	rows, ok := value.([]any)
	if !ok || len(rows) == 0 {
		return nil, errors.New("EXTERNAL_STAGE_ROLE_CONTRACT_INVALID")
	}
	roles := make(map[string]struct{}, len(rows))
	for _, item := range rows {
		role, ok := item.(string)
		if !ok || role == "" || role != strings.TrimSpace(role) {
			return nil, errors.New("EXTERNAL_STAGE_ROLE_CONTRACT_INVALID")
		}
		if _, exists := roles[role]; exists {
			return nil, errors.New("EXTERNAL_STAGE_ROLE_CONTRACT_INVALID")
		}
		roles[role] = struct{}{}
	}
	return roles, nil
}

func Bind(ctx context.Context, pool *pgxpool.Pool, req BindRequest) (Bound, error) {
	req.StageID = strings.TrimSpace(req.StageID)
	req.BoundBy = strings.TrimSpace(req.BoundBy)
	if req.AttemptID == uuid.Nil || req.StageID == "" || req.BoundBy == "" || len(req.Bindings) == 0 {
		return Bound{}, errors.New("external evidence binding identity is incomplete")
	}
	bindings := append([]Binding(nil), req.Bindings...)
	for i := range bindings {
		bindings[i].Role = strings.TrimSpace(bindings[i].Role)
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].Role < bindings[j].Role })
	seen := map[string]struct{}{}
	material := make([]map[string]string, 0, len(bindings))
	for _, b := range bindings {
		if b.Role == "" || b.ArtifactID == uuid.Nil {
			return Bound{}, errors.New("external evidence role and artifact are required")
		}
		if _, ok := seen[b.Role]; ok {
			return Bound{}, errors.New("EXTERNAL_STAGE_EVIDENCE_ROLE_DUPLICATE")
		}
		seen[b.Role] = struct{}{}
		material = append(material, map[string]string{"role": b.Role, "artifact_id": b.ArtifactID.String()})
	}
	evidenceSHA, err := semantic.JSONSHA256(map[string]any{"schema": "RealSaS.ExternalStageEvidenceBinding.v1", "attempt_id": req.AttemptID.String(), "stage_id": req.StageID, "bindings": material})
	if err != nil {
		return Bound{}, err
	}
	reused := true
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var state, kind string
		var releaseID uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT final_state,engine_release_id,kind FROM attempts WHERE id=$1 FOR UPDATE", req.AttemptID).Scan(&state, &releaseID, &kind); err != nil {
			return err
		}
		if state != "OPEN" {
			return errors.New("EXTERNAL_STAGE_ATTEMPT_NOT_OPEN")
		}
		if kind != "research" {
			return errors.New("EXTERNAL_STAGE_RESEARCH_ATTEMPT_REQUIRED")
		}
		var parametersRaw []byte
		if err := tx.QueryRow(ctx, "SELECT semantic_parameters FROM engine_release_stages WHERE release_id=$1 AND stage_id=$2", releaseID, req.StageID).Scan(&parametersRaw); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return errors.New("EXTERNAL_STAGE_NOT_IN_RELEASE")
			}
			return err
		}
		allowedRoles, err := declaredExternalRoles(parametersRaw)
		if err != nil {
			return err
		}
		for _, b := range bindings {
			if _, ok := allowedRoles[b.Role]; !ok {
				return errors.New("EXTERNAL_STAGE_ROLE_NOT_RELEASE_AUTHORIZED")
			}
			var producer string
			if err := tx.QueryRow(ctx, "SELECT producer_contract FROM artifacts WHERE id=$1", b.ArtifactID).Scan(&producer); err != nil {
				return err
			}
			if producer != "RealSaS.ExternalEvidenceImport.v1" {
				return errors.New("EXTERNAL_STAGE_ARTIFACT_NOT_RAW_IMPORT")
			}
			var existing uuid.UUID
			err := tx.QueryRow(ctx, "SELECT artifact_id FROM external_stage_evidence WHERE attempt_id=$1 AND stage_id=$2 AND role=$3 FOR UPDATE", req.AttemptID, req.StageID, b.Role).Scan(&existing)
			if err == nil {
				if existing != b.ArtifactID {
					return errors.New("EXTERNAL_STAGE_EVIDENCE_DRIFT")
				}
				continue
			}
			if !errors.Is(err, pgx.ErrNoRows) {
				return err
			}
			reused = false
			if _, err := tx.Exec(ctx, `INSERT INTO external_stage_evidence(attempt_id,stage_id,role,artifact_id,bound_by) VALUES ($1,$2,$3,$4,$5)`, req.AttemptID, req.StageID, b.Role, b.ArtifactID, req.BoundBy); err != nil {
				return err
			}
		}
		raw, _ := json.Marshal(map[string]any{"schema": "RealSaS.ExternalStageEvidenceBinding.v1", "stage_id": req.StageID, "evidence_sha256": evidenceSHA, "roles": material})
		_, err = tx.Exec(ctx, `INSERT INTO attempt_events(attempt_id,event_type,payload) SELECT $1,'EXTERNAL_STAGE_EVIDENCE_BOUND',$2::jsonb WHERE NOT EXISTS (SELECT 1 FROM attempt_events WHERE attempt_id=$1 AND event_type='EXTERNAL_STAGE_EVIDENCE_BOUND' AND payload=$2::jsonb)`, req.AttemptID, raw)
		return err
	})
	if err != nil {
		return Bound{}, err
	}

	var workflowID, status string
	signal := false
	err = pool.QueryRow(ctx, `SELECT workflow_id,status FROM executions WHERE attempt_id=$1 AND stage_contract=$2`, req.AttemptID, req.StageID).Scan(&workflowID, &status)
	if err == nil && status == "WAITING_EXTERNAL_EVIDENCE" {
		_, err = command.SubmitExternalStageReady(ctx, pool, command.ExternalStageReadyRequest{
			AttemptID: req.AttemptID, StageID: req.StageID, WorkflowID: workflowID, EvidenceSHA: evidenceSHA,
			IdempotencyKey: "external-stage-ready:" + req.AttemptID.String() + ":" + req.StageID + ":" + evidenceSHA, RequestedBy: req.BoundBy,
		})
		if err != nil {
			return Bound{}, err
		}
		signal = true
	} else if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return Bound{}, err
	}
	return Bound{AttemptID: req.AttemptID, StageID: req.StageID, EvidenceSHA256: evidenceSHA, WorkflowSignaled: signal, Reused: reused}, nil
}
