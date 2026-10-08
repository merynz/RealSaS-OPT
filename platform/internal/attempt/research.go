package attempt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

var (
	ErrSubjectNotFound       = errors.New("research subject not found")
	ErrParentSubjectMismatch = errors.New("research parent attempt subject mismatch")
	ErrCandidateNotResearch  = errors.New("research attempt requires RESEARCH release")
)

type ResearchRequest struct {
	SubjectID                uuid.UUID  `json:"subject_id"`
	BaselineEngineReleaseID  uuid.UUID  `json:"baseline_engine_release_id"`
	CandidateEngineReleaseID uuid.UUID  `json:"candidate_engine_release_id"`
	ParentAttemptID          *uuid.UUID `json:"parent_attempt_id"`
	CreatedBy                string     `json:"created_by"`
}

type ResearchStart struct {
	AttemptID                uuid.UUID
	BaselineEngineReleaseID  uuid.UUID
	CandidateEngineReleaseID uuid.UUID
	Impact                   release.Impact
}

func StartResearch(
	ctx context.Context,
	pool *pgxpool.Pool,
	g *stagegraph.Graph,
	req ResearchRequest,
) (ResearchStart, error) {
	if req.SubjectID == uuid.Nil || req.BaselineEngineReleaseID == uuid.Nil || req.CandidateEngineReleaseID == uuid.Nil {
		return ResearchStart{}, errors.New("subject/baseline/candidate release ids are required")
	}
	if req.CreatedBy == "" {
		return ResearchStart{}, errors.New("created_by is required")
	}

	baselineInfo, err := release.LoadInfo(ctx, pool, req.BaselineEngineReleaseID)
	if err != nil {
		return ResearchStart{}, err
	}
	candidateInfo, err := release.LoadInfo(ctx, pool, req.CandidateEngineReleaseID)
	if err != nil {
		return ResearchStart{}, err
	}
	if candidateInfo.Purpose != release.PurposeResearch {
		return ResearchStart{}, ErrCandidateNotResearch
	}
	baselineVersions, err := release.LoadVersions(ctx, pool, req.BaselineEngineReleaseID, false)
	if err != nil {
		return ResearchStart{}, err
	}
	candidateVersions, err := release.LoadVersions(ctx, pool, req.CandidateEngineReleaseID, false)
	if err != nil {
		return ResearchStart{}, err
	}
	impact, err := release.Compare(g, baselineVersions, candidateVersions)
	if err != nil {
		return ResearchStart{}, err
	}

	directIDs := make([]string, 0, len(impact.DirectChanges))
	for _, change := range impact.DirectChanges {
		directIDs = append(directIDs, change.StageID)
	}
	spec := map[string]any{
		"schema":                      "RealSaS.ResearchAttemptSpec.v1",
		"subject_id":                  req.SubjectID.String(),
		"baseline_engine_release_id":  req.BaselineEngineReleaseID.String(),
		"baseline_release_sha256":     baselineInfo.ReleaseSHA256,
		"candidate_engine_release_id": req.CandidateEngineReleaseID.String(),
		"candidate_release_sha256":    candidateInfo.ReleaseSHA256,
		"direct_changed_stage_ids":    directIDs,
		"invalidated_stage_ids":       impact.InvalidatedStageIDs,
	}
	if req.ParentAttemptID != nil {
		spec["parent_attempt_id"] = req.ParentAttemptID.String()
	}
	specSHA, err := semantic.JSONSHA256(spec)
	if err != nil {
		return ResearchStart{}, err
	}

	attemptID := uuid.New()
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var subject uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR SHARE", req.SubjectID).Scan(&subject); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrSubjectNotFound
			}
			return err
		}
		if req.ParentAttemptID != nil {
			var parentSubject uuid.UUID
			if err := tx.QueryRow(ctx, "SELECT subject_id FROM attempts WHERE id=$1", *req.ParentAttemptID).Scan(&parentSubject); err != nil {
				if errors.Is(err, pgx.ErrNoRows) {
					return ErrParentSubjectMismatch
				}
				return err
			}
			if parentSubject != req.SubjectID {
				return ErrParentSubjectMismatch
			}
		}

		if _, err := tx.Exec(ctx, `
			INSERT INTO attempts
			  (id,subject_id,engine_release_id,parent_attempt_id,kind,spec_sha256,created_by,final_state)
			VALUES ($1,$2,$3,$4,'research',$5,$6,'OPEN')
		`, attemptID, req.SubjectID, req.CandidateEngineReleaseID, req.ParentAttemptID, specSHA, req.CreatedBy); err != nil {
			return err
		}

		payload, err := json.Marshal(map[string]any{
			"schema":                      "RealSaS.CodeChangeImpact.v1",
			"baseline_engine_release_id":  req.BaselineEngineReleaseID.String(),
			"candidate_engine_release_id": req.CandidateEngineReleaseID.String(),
			"direct_changes":              impact.DirectChanges,
			"direct_changed_stage_ids":    directIDs,
			"invalidated_stage_ids":       impact.InvalidatedStageIDs,
			"unchanged_stage_ids":         impact.UnchangedStageIDs,
		})
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO attempt_events(attempt_id,event_type,payload)
			VALUES ($1,'CODE_CHANGE_IMPACT_RESOLVED',$2)
		`, attemptID, payload); err != nil {
			return err
		}

		audit, _ := json.Marshal(map[string]any{
			"attempt_id":                  attemptID.String(),
			"baseline_engine_release_id":  req.BaselineEngineReleaseID.String(),
			"candidate_engine_release_id": req.CandidateEngineReleaseID.String(),
			"direct_changed_stage_ids":    directIDs,
			"invalidated_stage_count":     len(impact.InvalidatedStageIDs),
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,attempt_id,payload)
			VALUES ($1,'RESEARCH_ATTEMPT_STARTED',$2,$3,$4)
		`, req.CreatedBy, req.SubjectID, attemptID, audit); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return ResearchStart{}, fmt.Errorf("start research attempt: %w", err)
	}
	return ResearchStart{
		AttemptID:                attemptID,
		BaselineEngineReleaseID:  req.BaselineEngineReleaseID,
		CandidateEngineReleaseID: req.CandidateEngineReleaseID,
		Impact:                   impact,
	}, nil
}
