package capability

import (
	"errors"
	"fmt"
)

type GoalType string

const (
	GoalProductCompile GoalType = "PRODUCT_COMPILE"
	GoalDeveloperRun   GoalType = "DEVELOPER_RUN"
	GoalProofRun       GoalType = "PROOF_RUN"
)

type PromotionPolicy string

const (
	PromotionNever                PromotionPolicy = "NEVER"
	PromotionQualifiedProductOnly PromotionPolicy = "QUALIFIED_PRODUCT_ONLY"
)

type Goal struct {
	Type            GoalType        `json:"type"`
	Targets         []string        `json:"targets"`
	PromotionPolicy PromotionPolicy `json:"promotion_policy"`
}

func (g Goal) Validate(registry *Registry) ([]Descriptor, error) {
	if registry == nil {
		return nil, errors.New("goal requires capability registry")
	}
	if len(g.Targets) == 0 {
		return nil, errors.New("goal requires at least one target")
	}
	switch g.Type {
	case GoalProductCompile:
		if g.PromotionPolicy != PromotionQualifiedProductOnly {
			return nil, errors.New("product compile must use qualified-product promotion policy")
		}
	case GoalDeveloperRun, GoalProofRun:
		if g.PromotionPolicy != PromotionNever {
			return nil, fmt.Errorf("%s cannot promote product state", g.Type)
		}
	default:
		return nil, fmt.Errorf("unsupported goal type %q", g.Type)
	}
	resolved, err := registry.Resolve(g.Targets...)
	if err != nil {
		return nil, err
	}
	if g.Type == GoalProductCompile {
		if len(g.Targets) != 1 {
			return nil, errors.New("product compile requires exactly one terminal target")
		}
		target, _ := registry.Get(g.Targets[0])
		if !target.ProductPassAuthority {
			return nil, fmt.Errorf("product target %s is not product-pass authority", target.ID)
		}
	}
	mode := ModeDeveloper
	if g.Type == GoalProductCompile {
		mode = ModeProduct
	}
	for _, descriptor := range resolved {
		allowed := false
		for _, candidate := range descriptor.AllowedModes {
			if candidate == mode {
				allowed = true
				break
			}
		}
		if !allowed {
			return nil, fmt.Errorf("capability %s is not allowed in %s mode", descriptor.ID, mode)
		}
	}
	return resolved, nil
}
