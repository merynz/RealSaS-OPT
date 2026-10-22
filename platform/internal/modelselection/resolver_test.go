package modelselection

import (
	"strings"
	"testing"

	"github.com/google/uuid"
)

func resolverSelection(family, checkpointSHA string) Selection {
	return Selection{
		Family:                  family,
		ModelReleaseID:          uuid.New(),
		ArchitectureID:          "RealSaS.TestArchitecture.v1",
		ModelEngineReleaseID:    uuid.New(),
		SourceCodeSHA:           strings.Repeat("a", 40),
		CheckpointArtifactID:    uuid.New(),
		CheckpointContentSHA256: checkpointSHA,
		Channel:                 "PRODUCTION",
		ChannelLockVersion:      7,
	}
}

func TestResolverInputsAreDeterministicAndChannelProvenanceIsNotBehaviorIdentity(t *testing.T) {
	a := resolverSelection("IRIS", strings.Repeat("1", 64))
	b := resolverSelection("TESSA", strings.Repeat("2", 64))

	forward, err := ResolverInputsFromSelections([]Selection{a, b})
	if err != nil {
		t.Fatal(err)
	}
	reverse, err := ResolverInputsFromSelections([]Selection{b, a})
	if err != nil {
		t.Fatal(err)
	}
	if len(forward) != 2 || len(reverse) != 2 {
		t.Fatalf("unexpected resolver input count: %d %d", len(forward), len(reverse))
	}
	for i := range forward {
		if forward[i] != reverse[i] {
			t.Fatalf("model resolver ordering is not canonical: %+v %+v", forward, reverse)
		}
	}
	if forward[0].Role != "model:IRIS" || forward[1].Role != "model:TESSA" {
		t.Fatalf("unexpected model roles: %+v", forward)
	}

	moved := a
	moved.Channel = "CANDIDATE"
	moved.ChannelLockVersion = 999
	before, err := BehaviorIdentitySHA256(a)
	if err != nil {
		t.Fatal(err)
	}
	after, err := BehaviorIdentitySHA256(moved)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("channel provenance incorrectly changed immutable behavior identity")
	}
}

func TestResolverIdentityChangesForExactModelOrCheckpointChange(t *testing.T) {
	base := resolverSelection("IRIS", strings.Repeat("3", 64))
	baseSHA, err := BehaviorIdentitySHA256(base)
	if err != nil {
		t.Fatal(err)
	}

	changedRelease := base
	changedRelease.ModelReleaseID = uuid.New()
	releaseSHA, err := BehaviorIdentitySHA256(changedRelease)
	if err != nil {
		t.Fatal(err)
	}
	if releaseSHA == baseSHA {
		t.Fatal("different ModelRelease aliased behavior identity")
	}

	changedCheckpoint := base
	changedCheckpoint.CheckpointContentSHA256 = strings.Repeat("4", 64)
	checkpointSHA, err := BehaviorIdentitySHA256(changedCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	if checkpointSHA == baseSHA {
		t.Fatal("different checkpoint bytes aliased behavior identity")
	}
}

func TestResolverInputsRejectDuplicateOrUnsafeFamilies(t *testing.T) {
	a := resolverSelection("IRIS", strings.Repeat("5", 64))
	b := resolverSelection("IRIS", strings.Repeat("6", 64))
	if _, err := ResolverInputsFromSelections([]Selection{a, b}); err == nil || err.Error() != "MODEL_SELECTION_DUPLICATE_FAMILY" {
		t.Fatalf("duplicate family admitted: %v", err)
	}

	unsafe := resolverSelection("../IRIS", strings.Repeat("7", 64))
	if _, err := ResolverInputsFromSelections([]Selection{unsafe}); err == nil || err.Error() != "MODEL_SELECTION_FAMILY_INVALID" {
		t.Fatalf("unsafe family admitted: %v", err)
	}
}
