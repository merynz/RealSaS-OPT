package campaign

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

const (
	RoleCohort     = "COHORT"
	RoleRegression = "REGRESSION"
)

type StartRequest struct {
	Name                     string      `json:"name"`
	BaselineEngineReleaseID  uuid.UUID   `json:"baseline_engine_release_id"`
	CandidateEngineReleaseID uuid.UUID   `json:"candidate_engine_release_id"`
	TargetStageID            string      `json:"target_stage_id,omitempty"`
	CohortSubjectIDs         []uuid.UUID `json:"cohort_subject_ids"`
	RegressionSubjectIDs     []uuid.UUID `json:"regression_subject_ids,omitempty"`
	CreatedBy                string      `json:"created_by"`
}

type Member struct {
	SubjectID  uuid.UUID `json:"subject_id"`
	AttemptID  uuid.UUID `json:"attempt_id"`
	Role       string    `json:"role"`
	Ordinal    int       `json:"ordinal"`
	FinalState string    `json:"final_state"`
}

type Status struct {
	CampaignID               uuid.UUID `json:"campaign_id"`
	Name                     string    `json:"name"`
	BaselineEngineReleaseID  uuid.UUID `json:"baseline_engine_release_id"`
	CandidateEngineReleaseID uuid.UUID `json:"candidate_engine_release_id"`
	TargetStageID            string    `json:"target_stage_id"`
	CanonicalCodeSHA         string    `json:"canonical_code_sha"`
	State                    string    `json:"state"`
	SpecSHA256               string    `json:"spec_sha256"`
	Members                  []Member  `json:"members"`
	CohortComplete           bool      `json:"cohort_complete"`
	RegressionComplete       bool      `json:"regression_complete"`
	FullClosureComplete      bool      `json:"full_closure_complete"`
	GoalComplete             bool      `json:"goal_complete"`
	ExecutionScope           string    `json:"execution_scope"`
	AnyFailed                bool      `json:"any_failed"`
	Fit8PassClaimed          bool      `json:"fit8_pass_claimed"`
	CreatedAt                time.Time `json:"created_at"`
}

type MemberRunBinding struct {
	SubjectID       uuid.UUID `json:"subject_id"`
	SubjectInputID  uuid.UUID `json:"subject_input_id"`
	RunManifestPath string    `json:"run_manifest_path"`
	RunLedgerPath   string    `json:"run_ledger_path"`
}

type RunRequest struct {
	CampaignID         uuid.UUID          `json:"campaign_id"`
	Members            []MemberRunBinding `json:"members"`
	PipelinePlanSHA256 string             `json:"pipeline_plan_sha256"`
	RequestedBy        string             `json:"requested_by"`
}

type RunReceipt struct {
	CampaignID uuid.UUID         `json:"campaign_id"`
	Receipts   []command.Receipt `json:"receipts"`
	State      string            `json:"state"`
}

type stored struct {
	ID                       uuid.UUID
	Name                     string
	BaselineEngineReleaseID  uuid.UUID
	CandidateEngineReleaseID uuid.UUID
	TargetStageID            string
	CanonicalCodeSHA         string
	State                    string
	SpecSHA256               string
	CreatedAt                time.Time
}

func orderedMembers(req StartRequest) ([]struct {
	SubjectID uuid.UUID
	Role      string
	Ordinal   int
}, error) {
	if len(req.CohortSubjectIDs) == 0 {
		return nil, errors.New("RESEARCH_CAMPAIGN_COHORT_REQUIRED")
	}
	seen := map[uuid.UUID]struct{}{}
	rows := make([]struct {
		SubjectID uuid.UUID
		Role      string
		Ordinal   int
	}, 0, len(req.CohortSubjectIDs)+len(req.RegressionSubjectIDs))
	appendRole := func(ids []uuid.UUID, role string) error {
		for i, id := range ids {
			if id == uuid.Nil {
				return errors.New("RESEARCH_CAMPAIGN_SUBJECT_ID_REQUIRED")
			}
			if _, ok := seen[id]; ok {
				return errors.New("RESEARCH_CAMPAIGN_DUPLICATE_SUBJECT")
			}
			seen[id] = struct{}{}
			rows = append(rows, struct {
				SubjectID uuid.UUID
				Role      string
				Ordinal   int
			}{id, role, i})
		}
		return nil
	}
	if err := appendRole(req.CohortSubjectIDs, RoleCohort); err != nil {
		return nil, err
	}
	if err := appendRole(req.RegressionSubjectIDs, RoleRegression); err != nil {
		return nil, err
	}
	return rows, nil
}

// campaignTarget preserves the default whole-product contract while allowing an
// explicit research component to finish without entering downstream external waits.
func campaignTarget(graph *stagegraph.Graph, requested string) (string, string, error) {
	product, hasProduct := graph.ProductPassStageID()
	target := strings.TrimSpace(requested)
	if target == "" {
		if !hasProduct {
			return "", "", errors.New("RESEARCH_CAMPAIGN_PRODUCT_CLOSURE_STAGE_REQUIRED")
		}
		target = product
	}
	closure, err := graph.AncestorsIncluding(target)
	if err != nil {
		return "", "", err
	}
	if hasProduct && target == product {
		if len(closure) != graph.StageCount() {
			return "", "", errors.New("RESEARCH_CAMPAIGN_PRODUCT_TARGET_MUST_CLOSE_FULL_DAG")
		}
		return target, "FULL_DAG_FROM_SOURCE_ENTRY_TO_PRODUCT_CLOSURE", nil
	}
	// A component campaign retains every declared source-entry root. It cannot
	// silently substitute an easier input lane or start after source qualification.
	parents := make(map[string]bool, len(closure))
	for _, id := range closure {
		parents[id] = true
	}
	for _, stage := range graph.Stages() {
		if len(stage.DependsOn) == 0 && !parents[stage.ID] {
			return "", "", errors.New("RESEARCH_CAMPAIGN_COMPONENT_SOURCE_ENTRY_MISSING")
		}
	}
	return target, "SOURCE_ENTRY_TO_RESEARCH_COMPONENT", nil
}

func Start(ctx context.Context, pool *pgxpool.Pool, apiGraph *stagegraph.Graph, deployedCodeSHA string, req StartRequest) (Status, error) {
	req.Name = strings.TrimSpace(req.Name)
	req.CreatedBy = strings.TrimSpace(req.CreatedBy)
	if req.Name == "" || req.CreatedBy == "" || req.BaselineEngineReleaseID == uuid.Nil || req.CandidateEngineReleaseID == uuid.Nil {
		return Status{}, errors.New("RESEARCH_CAMPAIGN_IDENTITY_REQUIRED")
	}
	if len(deployedCodeSHA) != 40 {
		return Status{}, errors.New("RESEARCH_CAMPAIGN_CANONICAL_CODE_REQUIRED")
	}
	members, err := orderedMembers(req)
	if err != nil {
		return Status{}, err
	}
	if err := release.RequireCanonicalMain(ctx, pool, req.CandidateEngineReleaseID, deployedCodeSHA); err != nil {
		return Status{}, err
	}
	candidateGraph, _, err := release.LoadGraph(ctx, pool, req.CandidateEngineReleaseID)
	if err != nil {
		return Status{}, err
	}
	target, scope, err := campaignTarget(candidateGraph, req.TargetStageID)
	if err != nil {
		return Status{}, err
	}
	specMembers := make([]map[string]any, 0, len(members))
	for _, row := range members {
		specMembers = append(specMembers, map[string]any{"subject_id": row.SubjectID.String(), "role": row.Role, "ordinal": row.Ordinal})
	}
	spec := map[string]any{
		"schema":                      "RealSaS.ResearchCampaignSpec.v1",
		"name":                        req.Name,
		"baseline_engine_release_id":  req.BaselineEngineReleaseID.String(),
		"candidate_engine_release_id": req.CandidateEngineReleaseID.String(),
		"target_stage_id":             target,
		"execution_scope":             scope,
		"canonical_code_sha":          deployedCodeSHA,
		"members":                     specMembers,
	}
	specSHA, err := semantic.JSONSHA256(spec)
	if err != nil {
		return Status{}, err
	}
	var existing uuid.UUID
	err = pool.QueryRow(ctx, "SELECT id FROM research_campaigns WHERE spec_sha256=$1", specSHA).Scan(&existing)
	if err == nil {
		return Get(ctx, pool, existing)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Status{}, err
	}

	campaignID := uuid.New()
	if _, err := pool.Exec(ctx, `INSERT INTO research_campaigns
        (id,name,baseline_engine_release_id,candidate_engine_release_id,target_stage_id,canonical_code_sha,spec_sha256,state,created_by)
        VALUES ($1,$2,$3,$4,$5,$6,$7,'STARTING',$8)`,
		campaignID, req.Name, req.BaselineEngineReleaseID, req.CandidateEngineReleaseID, target, deployedCodeSHA, specSHA, req.CreatedBy); err != nil {
		return Status{}, err
	}
	recordEvent(ctx, pool, campaignID, "CAMPAIGN_STARTING", spec)

	for _, row := range members {
		started, err := attempt.StartResearch(ctx, pool, apiGraph, attempt.ResearchRequest{
			SubjectID:                row.SubjectID,
			BaselineEngineReleaseID:  req.BaselineEngineReleaseID,
			CandidateEngineReleaseID: req.CandidateEngineReleaseID,
			CreatedBy:                req.CreatedBy,
			DeployedCodeSHA:          deployedCodeSHA,
		})
		if err != nil {
			_, _ = pool.Exec(ctx, "UPDATE research_campaigns SET state='START_FAILED' WHERE id=$1", campaignID)
			recordEvent(ctx, pool, campaignID, "CAMPAIGN_MEMBER_START_FAILED", map[string]any{"subject_id": row.SubjectID, "role": row.Role, "error": err.Error()})
			return Status{}, fmt.Errorf("start campaign member %s: %w", row.SubjectID, err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO research_campaign_members(campaign_id,subject_id,attempt_id,member_role,ordinal)
            VALUES ($1,$2,$3,$4,$5)`, campaignID, row.SubjectID, started.AttemptID, row.Role, row.Ordinal); err != nil {
			_, _ = pool.Exec(ctx, "UPDATE research_campaigns SET state='START_FAILED' WHERE id=$1", campaignID)
			return Status{}, err
		}
	}
	if _, err := pool.Exec(ctx, "UPDATE research_campaigns SET state='READY' WHERE id=$1", campaignID); err != nil {
		return Status{}, err
	}
	recordEvent(ctx, pool, campaignID, "CAMPAIGN_READY", map[string]any{"member_count": len(members), "target_stage_id": target})
	return Get(ctx, pool, campaignID)
}

func Get(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (Status, error) {
	if id == uuid.Nil {
		return Status{}, errors.New("RESEARCH_CAMPAIGN_ID_REQUIRED")
	}
	var c stored
	if err := pool.QueryRow(ctx, `SELECT id,name,baseline_engine_release_id,candidate_engine_release_id,target_stage_id,
        canonical_code_sha,state,spec_sha256,created_at FROM research_campaigns WHERE id=$1`, id).
		Scan(&c.ID, &c.Name, &c.BaselineEngineReleaseID, &c.CandidateEngineReleaseID, &c.TargetStageID,
			&c.CanonicalCodeSHA, &c.State, &c.SpecSHA256, &c.CreatedAt); err != nil {
		return Status{}, err
	}
	graph, _, err := release.LoadGraph(ctx, pool, c.CandidateEngineReleaseID)
	if err != nil {
		return Status{}, err
	}
	_, scope, err := campaignTarget(graph, c.TargetStageID)
	if err != nil {
		return Status{}, err
	}
	rows, err := pool.Query(ctx, `SELECT m.subject_id,m.attempt_id,m.member_role,m.ordinal,a.final_state
        FROM research_campaign_members m JOIN attempts a ON a.id=m.attempt_id
        WHERE m.campaign_id=$1 ORDER BY CASE m.member_role WHEN 'COHORT' THEN 0 ELSE 1 END,m.ordinal`, id)
	if err != nil {
		return Status{}, err
	}
	defer rows.Close()
	out := Status{
		CampaignID: c.ID, Name: c.Name,
		BaselineEngineReleaseID:  c.BaselineEngineReleaseID,
		CandidateEngineReleaseID: c.CandidateEngineReleaseID,
		TargetStageID:            c.TargetStageID, CanonicalCodeSHA: c.CanonicalCodeSHA,
		State: c.State, SpecSHA256: c.SpecSHA256, CreatedAt: c.CreatedAt,
		Fit8PassClaimed: false,
		ExecutionScope:  scope,
	}
	cohortTotal, cohortComplete := 0, 0
	regressionTotal, regressionComplete := 0, 0
	for rows.Next() {
		var m Member
		if err := rows.Scan(&m.SubjectID, &m.AttemptID, &m.Role, &m.Ordinal, &m.FinalState); err != nil {
			return Status{}, err
		}
		out.Members = append(out.Members, m)
		if m.FinalState == "FAILED" {
			out.AnyFailed = true
		}
		if m.Role == RoleCohort {
			cohortTotal++
			if m.FinalState == "COMPLETED" {
				cohortComplete++
			}
		} else if m.Role == RoleRegression {
			regressionTotal++
			if m.FinalState == "COMPLETED" {
				regressionComplete++
			}
		}
	}
	if err := rows.Err(); err != nil {
		return Status{}, err
	}
	out.CohortComplete = cohortTotal > 0 && cohortComplete == cohortTotal
	out.RegressionComplete = regressionComplete == regressionTotal
	out.GoalComplete = out.CohortComplete && out.RegressionComplete && !out.AnyFailed
	out.FullClosureComplete = out.GoalComplete && scope == "FULL_DAG_FROM_SOURCE_ENTRY_TO_PRODUCT_CLOSURE"
	return out, nil
}

func Run(ctx context.Context, pool *pgxpool.Pool, graph *stagegraph.Graph, deployedCodeSHA string, req RunRequest) (RunReceipt, error) {
	if req.CampaignID == uuid.Nil || strings.TrimSpace(req.RequestedBy) == "" {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_RUN_IDENTITY_REQUIRED")
	}
	if err := semantic.ValidateSHA256(req.PipelinePlanSHA256); err != nil {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_PIPELINE_SHA_REQUIRED")
	}
	status, err := Get(ctx, pool, req.CampaignID)
	if err != nil {
		return RunReceipt{}, err
	}
	if status.State != "READY" && status.State != "RUNNING" {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_NOT_RUNNABLE")
	}
	if status.CanonicalCodeSHA != deployedCodeSHA {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_CODE_DRIFT")
	}
	_, releasedPlanSHA, err := release.LoadGraph(ctx, pool, status.CandidateEngineReleaseID)
	if err != nil {
		return RunReceipt{}, err
	}
	if releasedPlanSHA != "" && releasedPlanSHA != req.PipelinePlanSHA256 {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_RELEASE_PLAN_DRIFT")
	}
	attempts := map[uuid.UUID]uuid.UUID{}
	for _, m := range status.Members {
		attempts[m.SubjectID] = m.AttemptID
	}
	if len(req.Members) != len(attempts) {
		return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_MEMBER_BINDING_INCOMPLETE")
	}
	bindings := append([]MemberRunBinding(nil), req.Members...)
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].SubjectID.String() < bindings[j].SubjectID.String() })
	seen := map[uuid.UUID]struct{}{}
	for _, b := range bindings {
		_, ok := attempts[b.SubjectID]
		if !ok || b.SubjectInputID == uuid.Nil || strings.TrimSpace(b.RunManifestPath) == "" || strings.TrimSpace(b.RunLedgerPath) == "" {
			return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_MEMBER_BINDING_INVALID")
		}
		if _, dup := seen[b.SubjectID]; dup {
			return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_MEMBER_BINDING_DUPLICATE")
		}
		seen[b.SubjectID] = struct{}{}
		var inputSubject uuid.UUID
		if err := pool.QueryRow(ctx, "SELECT subject_id FROM subject_inputs WHERE id=$1", b.SubjectInputID).Scan(&inputSubject); err != nil {
			return RunReceipt{}, err
		}
		if inputSubject != b.SubjectID {
			return RunReceipt{}, errors.New("RESEARCH_CAMPAIGN_SUBJECT_INPUT_DRIFT")
		}
	}

	receipts := make([]command.Receipt, 0, len(bindings))
	for _, b := range bindings {
		attemptID := attempts[b.SubjectID]
		compilerRunID := fmt.Sprintf("CAMPAIGN_%s_%s", req.CampaignID.String(), b.SubjectID.String())
		idempotencyKey := fmt.Sprintf("campaign:%s:subject:%s:target:%s", req.CampaignID.String(), b.SubjectID.String(), status.TargetStageID)
		receipt, err := command.SubmitCompile(ctx, pool, graph, command.CompileRequest{
			ResearchAttemptID:  &attemptID,
			SubjectID:          b.SubjectID,
			EngineReleaseID:    status.CandidateEngineReleaseID,
			SubjectInputID:     b.SubjectInputID,
			TargetStageID:      status.TargetStageID,
			CompilerRunID:      compilerRunID,
			RunManifestPath:    b.RunManifestPath,
			RunLedgerPath:      b.RunLedgerPath,
			PipelinePlanSHA256: req.PipelinePlanSHA256,
			IdempotencyKey:     idempotencyKey,
			RequestedBy:        req.RequestedBy,
		})
		if err != nil {
			_, _ = pool.Exec(ctx, "UPDATE research_campaigns SET state='SUBMIT_FAILED' WHERE id=$1", req.CampaignID)
			recordEvent(ctx, pool, req.CampaignID, "CAMPAIGN_MEMBER_SUBMIT_FAILED", map[string]any{"subject_id": b.SubjectID, "error": err.Error()})
			return RunReceipt{}, err
		}
		receipts = append(receipts, receipt)
	}
	if _, err := pool.Exec(ctx, "UPDATE research_campaigns SET state='RUNNING' WHERE id=$1", req.CampaignID); err != nil {
		return RunReceipt{}, err
	}
	recordEvent(ctx, pool, req.CampaignID, "CAMPAIGN_COMPILE_SUBMITTED", map[string]any{"member_count": len(receipts), "target_stage_id": status.TargetStageID})
	return RunReceipt{CampaignID: req.CampaignID, Receipts: receipts, State: "RUNNING"}, nil
}

func recordEvent(ctx context.Context, pool *pgxpool.Pool, campaignID uuid.UUID, eventType string, payload any) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = pool.Exec(ctx, `INSERT INTO research_campaign_events(campaign_id,event_type,payload) VALUES ($1,$2,$3)`, campaignID, eventType, raw)
}
