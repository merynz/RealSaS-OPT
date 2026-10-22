package operational

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/experiment"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelrelease"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func repeated(ch string, n int) string {
	return strings.Repeat(ch, n)
}

func uniqueHash(prefix, suffix string) string {
	sum := sha256.Sum256([]byte(prefix + ":" + suffix))
	return hex.EncodeToString(sum[:])
}

func TestOperationalSpineMakesExperimentAndCurrentModelQueryable(t *testing.T) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	suffix := uuid.New().String()
	subjectID := uuid.New()
	releaseID := uuid.New()
	artifactTypeID := uuid.New()
	artifactID := uuid.New()
	attemptID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "ops-"+suffix, "Operational spine test"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())`, releaseID, "ops-release-"+suffix, uniqueHash("release", suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','model')", artifactTypeID, "ops-checkpoint-"+suffix); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,
		 implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'ci',$6,$7,'{}',now())`, artifactID, artifactTypeID,
		uniqueHash("semantic", suffix), uniqueHash("content", suffix), "ops/"+suffix,
		uniqueHash("implementation", suffix), uniqueHash("policy", suffix)); err != nil {
		t.Fatal(err)
	}

	created, err := experiment.Create(ctx, pool, experiment.CreateRequest{
		Slug:             "structural-evidence-" + suffix,
		Name:             "Structural Evidence A/B",
		ResearchQuestion: "Does role-free presentation evidence improve structuralization?",
		Hypothesis:       "Shared-scene presentation evidence improves topology and presentation behavior.",
		InitialState:     experiment.StatePreregistered,
		CreatedBy:        "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := experiment.Transition(ctx, pool, experiment.TransitionRequest{ExperimentID: created.Experiment.ID, ToState: experiment.StateRunning, Actor: "ci"}); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO attempts
		(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'research',$4,'ci','OPEN')`, attemptID, subjectID, releaseID, uniqueHash("attempt", suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := experiment.Link(ctx, pool, experiment.LinkRequest{ExperimentID: created.Experiment.ID, TargetKind: "ATTEMPT", TargetID: attemptID, Actor: "ci"}); err != nil {
		t.Fatal(err)
	}

	registered, err := modelrelease.Register(ctx, pool, modelrelease.RegisterRequest{
		Family:                  "IRIS",
		Name:                    "iris-ops-" + suffix,
		ArchitectureID:          "RealSaS.IRIS.Test.v1",
		CheckpointArtifactID:    artifactID,
		EngineReleaseID:         releaseID,
		OriginatingExperimentID: &created.Experiment.ID,
		OriginatingAttemptID:    &attemptID,
		SourceCodeSHA:           repeated("a", 40),
		InitialState:            modelrelease.StateResearchQualified,
		CreatedBy:               "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrelease.SetEvaluation(ctx, pool, modelrelease.EvaluationRequest{
		ModelReleaseID:     registered.ID,
		Scope:              "FIT8",
		Result:             "PASS",
		EvidenceArtifactID: &artifactID,
		ExperimentID:       &created.Experiment.ID,
		Actor:              "ci",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := modelrelease.Promote(ctx, pool, modelrelease.PromoteRequest{
		Family:                   "IRIS",
		Channel:                  modelrelease.ChannelCandidate,
		ModelReleaseID:           registered.ID,
		RequiredEvaluationScopes: []string{"FIT8"},
		Actor:                    "ci",
	}); err != nil {
		t.Fatal(err)
	}

	status, err := Status(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	foundExperiment := false
	for _, record := range status.ActiveExperiments {
		if record.ID == created.Experiment.ID {
			foundExperiment = true
		}
	}
	if !foundExperiment {
		t.Fatal("running experiment missing from operational status")
	}
	foundAttempt := false
	for _, record := range status.OpenAttempts {
		if record.ID == attemptID && record.ExperimentID != nil && *record.ExperimentID == created.Experiment.ID {
			foundAttempt = true
		}
	}
	if !foundAttempt {
		t.Fatal("experiment-linked attempt missing from operational status")
	}
	foundModel := false
	for _, record := range status.CurrentModels {
		if record.Channel.Family == "IRIS" && record.Channel.Channel == modelrelease.ChannelCandidate && record.Release.ID == registered.ID {
			foundModel = true
		}
	}
	if !foundModel {
		t.Fatal("candidate model missing from operational status")
	}
}
