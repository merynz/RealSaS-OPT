package attempt

import (
	"context"
	"github.com/google/uuid"
	"strings"
	"testing"
)

func TestParentContinuationCannotOptOutOfDeclaredIntervention(t *testing.T) {
	parent := uuid.New()
	_, err := StartResearch(context.Background(), nil, nil, ResearchRequest{ParentAttemptID: &parent})
	if err == nil || !strings.Contains(err.Error(), "CONTINUATION_REQUIRES_INTERVENTION") {
		t.Fatalf("%v", err)
	}
}
