package externalstage

import "testing"

func TestDeclaredExternalRolesSupportsRequiredAndOptionalReleaseContract(t *testing.T) {
	roles, err := declaredExternalRoles([]byte(`{"external_evidence_roles":["checkpoint","result","execution_receipt"],"optional_external_evidence_roles":["shared_execution_payload"]}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"checkpoint", "result", "execution_receipt", "shared_execution_payload"} {
		if _, ok := roles[role]; !ok {
			t.Fatalf("missing role %s", role)
		}
	}
	optionalOnly, err := declaredExternalRoles([]byte(`{"optional_external_evidence_roles":["diagnostic"]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := optionalOnly["diagnostic"]; !ok {
		t.Fatal("optional role missing")
	}
	for _, raw := range [][]byte{
		[]byte(`{}`),
		[]byte(`{"external_evidence_roles":[]}`),
		[]byte(`{"optional_external_evidence_roles":[]}`),
		[]byte(`{"external_evidence_roles":["checkpoint","checkpoint"]}`),
		[]byte(`{"optional_external_evidence_roles":[" checkpoint"]}`),
		[]byte(`{"external_evidence_roles":[1]}`),
		[]byte(`{"external_evidence_roles":["checkpoint"],"optional_external_evidence_roles":["checkpoint"]}`),
	} {
		if _, err := declaredExternalRoles(raw); err == nil {
			t.Fatalf("expected invalid contract rejection: %s", raw)
		}
	}
}
