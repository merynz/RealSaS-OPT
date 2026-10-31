package capability

import (
	"errors"
	"fmt"
)

type GoalType string

const (
	GoalProductCompile         GoalType = "PRODUCT_COMPILE"
	GoalProductRequalification GoalType = "PRODUCT_REQUALIFICATION"
	GoalDeveloperRun           GoalType = "DEVELOPER_RUN"
	GoalProofRun               GoalType = "PROOF_RUN"
)

type PromotionPolicy string

const (
	PromotionNever                PromotionPolicy = "NEVER"
	PromotionQualifiedProductOnly PromotionPolicy = "QUALIFIED_PRODUCT_ONLY"
)

type Goal struct {
	Type            GoalType        `json:"type"`
	Targets         []string        `json:"targets"`
	Satisfied       []string        `json:"satisfied,omitempty"`
	PromotionPolicy PromotionPolicy `json:"promotion_policy"`
}

func (g Goal) Validate(registry *Registry) ([]Descriptor, error) {
	if registry == nil {
		return nil, errors.New("goal requires capability registry")
	}
	if len(g.Targets) == 0 {
		return nil, errors.New("goal requires at least one target")
	}

	productGoal := false
	switch g.Type {
	case GoalProductCompile:
		productGoal = true
		if g.PromotionPolicy != PromotionQualifiedProductOnly {
			return nil, errors.New("product compile must use qualified-product promotion policy")
		}
		if len(g.Satisfied) != 0 {
			return nil, errors.New("product compile cannot declare satisfied capabilities")
		}
	case GoalProductRequalification:
		productGoal = true
		if g.PromotionPolicy != PromotionQualifiedProductOnly {
			return nil, errors.New("product requalification must use qualified-product promotion policy")
		}
		if len(g.Satisfied) == 0 {
			return nil, errors.New("product requalification requires satisfied base capabilities")
		}
	case GoalDeveloperRun, GoalProofRun:
		if g.PromotionPolicy != PromotionNever {
			return nil, fmt.Errorf("%s cannot promote product state", g.Type)
		}
		if len(g.Satisfied) != 0 {
			return nil, fmt.Errorf("%s cannot declare satisfied capabilities", g.Type)
		}
	default:
		return nil, fmt.Errorf("unsupported goal type %q", g.Type)
	}

	var (
		resolved []Descriptor
		err      error
	)
	if g.Type == GoalProductRequalification {
		resolved, err = registry.ResolveWithSatisfied(g.Targets, g.Satisfied)
	} else {
		resolved, err = registry.Resolve(g.Targets...)
	}
	if err != nil {
		return nil, err
	}

	if productGoal {
		if len(g.Targets) != 1 {
			return nil, errors.New("product goal requires exactly one terminal target")
		}
		target, _ := registry.Get(g.Targets[0])
		if !target.ProductPassAuthority {
			return nil, fmt.Errorf("product target %s is not product-pass authority", target.ID)
		}
	}

	mode := ModeDeveloper
	if productGoal {
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
	if productGoal {
		for _, id := range g.Satisfied {
			descriptor, ok := registry.Get(id)
			if !ok {
				return nil, fmt.Errorf("unknown satisfied capability %s", id)
			}
			allowed := false
			for _, candidate := range descriptor.AllowedModes {
				if candidate == ModeProduct {
					allowed = true
					break
				}
			}
			if !allowed {
				return nil, fmt.Errorf("satisfied capability %s is not allowed in PRODUCT mode", id)
			}
		}
	}
	return resolved, nil
}
