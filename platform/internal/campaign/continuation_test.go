package campaign

import (
	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"testing"
)

func TestContinuationRequiresExactTerminalCohortAndPerMemberIntervention(t *testing.T) {
	campaignID, releaseID, subjectID, attemptID := uuid.New(), uuid.New(), uuid.New(), uuid.New()
	parent := Status{CampaignID: campaignID, CandidateEngineReleaseID: releaseID, Members: []Member{{SubjectID: subjectID, AttemptID: attemptID, Role: RoleCohort, Ordinal: 0, FinalState: "FAILED"}}}
	req := StartRequest{ParentCampaignID: &campaignID, BaselineEngineReleaseID: releaseID, CohortSubjectIDs: []uuid.UUID{subjectID}, MemberInterventions: map[string]release.Intervention{subjectID.String(): {DirectChangedStageIDs: []string{"49_SUBJECT_TRUTH_FAMILIES"}}}}
	result, err := continuationParents(req, parent)
	if err != nil || result[subjectID] != attemptID {
		t.Fatalf("parents=%v err=%v", result, err)
	}
	parent.Members[0].FinalState = "OPEN"
	if _, err = continuationParents(req, parent); err == nil {
		t.Fatal("cannot silently restart active parent")
	}
	parent.Members[0].FinalState = "FAILED"
	delete(req.MemberInterventions, subjectID.String())
	if _, err = continuationParents(req, parent); err == nil {
		t.Fatal("explicit intervention required")
	}
}
