package modelrelease

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func TestPlatformPromotionPolicyIsSoleQualificationAuthority(t *testing.T) {
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
	family := "MIRA"
	architecture := "RealSaS.MIRA.PromotionPolicy.Test.v1"
	releaseID := uuid.New()
	artifactTypeID := uuid.New()
	artifactID := uuid.New()
	canonicalSHA := strings.Repeat("c", 40)
	checkpointSHA := sourceAuthorityHash("promotion-checkpoint:" + suffix)

	if _, err := pool.Exec(ctx, `INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())`, releaseID, "promotion-policy-"+suffix, sourceAuthorityHash("promotion-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := release.BindCanonicalMain(ctx, pool, releaseID, canonicalSHA, "ci"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','model')", artifactTypeID, "promotion-checkpoint-"+suffix); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,
		 implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'ci',$6,$7,'{}',now())`, artifactID, artifactTypeID,
		sourceAuthorityHash("promotion-semantic:"+suffix), checkpointSHA, "promotion-policy/"+suffix,
		sourceAuthorityHash("promotion-implementation:"+suffix), sourceAuthorityHash("promotion-artifact-policy:"+suffix)); err != nil {
		t.Fatal(err)
	}

	registered, err := Register(ctx, pool, RegisterRequest{
		Family:                  family,
		Name:                    "mira-promotion-policy-" + suffix,
		ArchitectureID:          architecture,
		CheckpointArtifactID:    artifactID,
		CheckpointContentSHA256: checkpointSHA,
		EngineReleaseID:         releaseID,
		SourceCodeSHA:           canonicalSHA,
		InitialState:            StateDraft,
		CreatedBy:               "ci",
	})
	if err != nil {
		t.Fatal(err)
	}

	// EASY is deliberately present and PASS, but it has no promotion authority.
	// Only the sealed Platform policy may define the qualification exam.
	if _, err := SetEvaluation(ctx, pool, EvaluationRequest{
		ModelReleaseID:     registered.ID,
		Scope:              "EASY",
		Result:             "PASS",
		EvidenceArtifactID: &artifactID,
		Actor:              "ci",
	}); err != nil {
		t.Fatal(err)
	}

	promote := func() error {
		_, err := Promote(ctx, pool, PromoteRequest{
			Family:         family,
			Channel:        ChannelCandidate,
			ModelReleaseID: registered.ID,
			Actor:          "ci",
		})
		return err
	}

	if err := promote(); err == nil || !strings.Contains(err.Error(), "MODEL_PROMOTION_POLICY_MISSING") {
		t.Fatalf("promotion without Platform policy was not rejected: %v", err)
	}

	policyID := uuid.New()
	policySHA := sourceAuthorityHash("promotion-policy-content:" + suffix)
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policies
		(id,family,architecture_id,channel,policy_version,required_evaluation_scopes,policy_sha256,sealed_by)
		VALUES ($1,$2,$3,'CANDIDATE',1,$4,$5,'ci')
	`, policyID, family, architecture, []string{"HARD"}, policySHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policy_bindings
		(family,architecture_id,channel,policy_id,bound_by)
		VALUES ($1,$2,'CANDIDATE',$3,'ci')
	`, family, architecture, policyID); err != nil {
		t.Fatal(err)
	}

	if err := promote(); err == nil || !strings.Contains(err.Error(), "MODEL_PROMOTION_POLICY_EVALUATION_MISSING:HARD") {
		t.Fatalf("decoy EASY PASS bypassed missing Platform HARD scope: %v", err)
	}

	if _, err := SetEvaluation(ctx, pool, EvaluationRequest{
		ModelReleaseID:     registered.ID,
		Scope:              "HARD",
		Result:             "FAIL",
		EvidenceArtifactID: &artifactID,
		Actor:              "ci",
	}); err != nil {
		t.Fatal(err)
	}
	if err := promote(); err == nil || !strings.Contains(err.Error(), "MODEL_PROMOTION_POLICY_EVALUATION_NOT_PASS:HARD:FAIL") {
		t.Fatalf("decoy EASY PASS bypassed failed Platform HARD scope: %v", err)
	}

	if _, err := pool.Exec(ctx, `UPDATE model_promotion_policies SET required_evaluation_scopes=$2 WHERE id=$1`, policyID, []string{"EASY"}); err == nil || !strings.Contains(err.Error(), "MODEL_PROMOTION_POLICY_IMMUTABLE") {
		t.Fatalf("sealed promotion policy was mutable: %v", err)
	}

	if _, err := SetEvaluation(ctx, pool, EvaluationRequest{
		ModelReleaseID:     registered.ID,
		Scope:              "HARD",
		Result:             "PASS",
		EvidenceArtifactID: &artifactID,
		Actor:              "ci",
	}); err != nil {
		t.Fatal(err)
	}
	channel, err := Promote(ctx, pool, PromoteRequest{
		Family:         family,
		Channel:        ChannelCandidate,
		ModelReleaseID: registered.ID,
		Actor:          "ci",
	})
	if err != nil {
		t.Fatalf("promotion was rejected after Platform-owned HARD scope passed: %v", err)
	}
	if channel.ModelReleaseID != registered.ID || channel.Channel != ChannelCandidate {
		t.Fatalf("unexpected candidate channel after policy-qualified promotion: %+v", channel)
	}
	mature, err := Get(ctx, pool, registered.ID)
	if err != nil {
		t.Fatal(err)
	}
	if mature.LifecycleState != StateProductCandidate {
		t.Fatalf("promotion did not advance lifecycle to PRODUCT_CANDIDATE: %s", mature.LifecycleState)
	}

	var auditJSON []byte
	if err := pool.QueryRow(ctx, `
		SELECT payload
		FROM audit_events
		WHERE action='MODEL_CHANNEL_PROMOTED'
		  AND payload->>'model_release_id'=$1
		ORDER BY id DESC
		LIMIT 1
	`, registered.ID.String()).Scan(&auditJSON); err != nil {
		t.Fatal(err)
	}
	var audit map[string]any
	if err := json.Unmarshal(auditJSON, &audit); err != nil {
		t.Fatal(err)
	}
	if audit["promotion_policy_id"] != policyID.String() {
		t.Fatalf("promotion audit lost policy id: %+v", audit)
	}
	if audit["promotion_policy_sha256"] != policySHA {
		t.Fatalf("promotion audit lost policy sha: %+v", audit)
	}
	scopes, ok := audit["promotion_policy_required_scopes"].([]any)
	if !ok || len(scopes) != 1 || scopes[0] != "HARD" {
		t.Fatalf("promotion audit lost policy scopes: %+v", audit)
	}
}
