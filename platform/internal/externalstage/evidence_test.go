package externalstage

import "testing"

func TestDeclaredExternalRolesRequiresExactUniqueReleaseContract(t *testing.T) {
	roles, err := declaredExternalRoles([]byte(`{"external_evidence_roles":["checkpoint","result","execution_receipt"]}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"checkpoint", "result", "execution_receipt"} {
		if _, ok := roles[role]; !ok {
			t.Fatalf("missing role %s", role)
		}
	}

	for _, raw := range [][]byte{
		[]byte(`{}`),
		[]byte(`{"external_evidence_roles":[]}`),
		[]byte(`{"external_evidence_roles":["checkpoint","checkpoint"]}`),
		[]byte(`{"external_evidence_roles":[" checkpoint"]}`),
		[]byte(`{"external_evidence_roles":[1]}`),
	} {
		if _, err := declaredExternalRoles(raw); err == nil {
			t.Fatalf("expected invalid contract rejection: %s", raw)
		}
	}
}
