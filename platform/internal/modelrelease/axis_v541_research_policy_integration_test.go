package modelrelease

import (
	"context"
	"os"
	"reflect"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestAXISV541ResearchPromotionPolicyIsNarrowAndSealed(t *testing.T) {
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

	const architecture = "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
	const expectedPolicySHA = "e715f018de5c03f5561989829f70a0cff8eeff0a692b2c73fc67de7300e7cfb9"
	expectedPolicyID := uuid.MustParse("9389310f-9555-5a42-ba35-a98f531ca725")
	expectedScopes := []string{
		"RUNTIME_PACKAGE_IDENTITY_V1",
		"RUNTIME_SELF_CONTAINED_LOAD_V1",
		"HISTORICAL_FIT1_TERMINAL_V541",
		"AXIS_RUNTIME_AUTHORITY_CONTRACT_V1",
	}

	var policyID uuid.UUID
	var policySHA string
	var version int64
	var scopes []string
	var bindingID uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT p.id,p.policy_sha256,p.policy_version,p.required_evaluation_scopes,b.policy_id
		FROM model_promotion_policy_bindings b
		JOIN model_promotion_policies p ON p.id=b.policy_id
		WHERE b.family='AXIS' AND b.architecture_id=$1 AND b.channel='RESEARCH'
	`, architecture).Scan(&policyID, &policySHA, &version, &scopes, &bindingID); err != nil {
		t.Fatal(err)
	}
	if policyID != expectedPolicyID || bindingID != expectedPolicyID {
		t.Fatalf("unexpected AXIS research policy identity: policy=%s binding=%s", policyID, bindingID)
	}
	if policySHA != expectedPolicySHA || version != 1 {
		t.Fatalf("unexpected AXIS research policy seal: sha=%s version=%d", policySHA, version)
	}
	if !reflect.DeepEqual(scopes, expectedScopes) {
		t.Fatalf("unexpected AXIS research policy scopes: %#v", scopes)
	}
	for _, forbidden := range []string{"FIT8", "LOFO", "UNSEEN_GENERALIZATION"} {
		for _, scope := range scopes {
			if scope == forbidden {
				t.Fatalf("AXIS RESEARCH policy improperly claims genericity scope %s", forbidden)
			}
		}
	}
}
