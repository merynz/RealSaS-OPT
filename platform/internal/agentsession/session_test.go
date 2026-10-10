package agentsession

import (
	"context"
	"strings"
	"testing"

	"github.com/google/uuid"
)

func TestCodeMismatchRejectsBeforeOpeningDatabase(t *testing.T) {
	for _, sha := range []string{"", strings.Repeat("z", 40), strings.Repeat("b", 40)} {
		_, err := Open(context.Background(), nil, strings.Repeat("a", 40), OpenRequest{CanonicalCodeSHA: sha})
		if err == nil || !strings.Contains(err.Error(), "CANONICAL_CODE_MISMATCH") {
			t.Fatalf("accepted %q: %v", sha, err)
		}
	}
}

func TestSessionScopeIsGenericAndRejectsEveryChangedBoundary(t *testing.T) {
	s := Scope{CanonicalLine: "main", AttemptID: uuid.New(), SubjectID: uuid.New(), EngineReleaseID: uuid.New(), SubjectInputID: uuid.New(), TargetStageID: "MODEL_NEW_HEAD_UNSEEN_EVALUATION"}
	if err := s.CheckCompile(s.AttemptID, s.SubjectID, s.EngineReleaseID, s.SubjectInputID, s.TargetStageID); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Scope){
		func(v *Scope) { v.CanonicalLine = "research" }, func(v *Scope) { v.AttemptID = uuid.New() }, func(v *Scope) { v.SubjectID = uuid.New() },
		func(v *Scope) { v.EngineReleaseID = uuid.New() }, func(v *Scope) { v.SubjectInputID = uuid.New() }, func(v *Scope) { v.TargetStageID = "OTHER_HEAD" },
	} {
		bad := s
		change(&bad)
		if err := bad.CheckCompile(s.AttemptID, s.SubjectID, s.EngineReleaseID, s.SubjectInputID, s.TargetStageID); err == nil {
			t.Fatalf("accepted %+v", bad)
		}
	}
}
