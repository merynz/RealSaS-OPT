package capability

import "sort"

func responsibilityIDsForRole(role ProductRoleID) ([]string, error) {
	ledger, err := loadResponsibilityLedger()
	if err != nil {
		return nil, err
	}
	var out []string
	for _, row := range ledger.Responsibilities {
		if row.RoleID == role && row.RequiredWhenRolePresent {
			out = append(out, row.ID)
		}
	}
	sort.Strings(out)
	return out, nil
}
