package attempt

import (
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func TestInterventionPinsInputsEvenWhenOnlyChangedStagesConsumeThem(t *testing.T) {
	source, rgb := uuid.New(), uuid.New()
	loaded := input.Loaded{RootInputs: []domain.ArtifactInputIdentity{{Role: "subject:source"}, {Role: "subject:manifest:appearance"}}, ArtifactIDs: []uuid.UUID{source, rgb}}
	c := &FrozenIntervention{BaselineInputs: PinInputs(loaded)}
	if err := c.CheckInputs(loaded); err != nil {
		t.Fatal(err)
	}
	loaded.ArtifactIDs[1] = uuid.New()
	if err := c.CheckInputs(loaded); err == nil {
		t.Fatal("undeclared RGB input accepted")
	}
	c.Contract = release.Intervention{ChangedInputRoles: []string{"manifest:appearance"}}
	if err := c.CheckInputs(loaded); err != nil {
		t.Fatal(err)
	}
	loaded.ArtifactIDs[0] = uuid.New()
	if err := c.CheckInputs(loaded); err == nil {
		t.Fatal("source drift hidden by allowed appearance change")
	}
	c.Contract.ChangedInputRoles = []string{"source", "manifest:appearance"}
	if err := c.CheckInputs(loaded); err != nil {
		t.Fatal(err)
	}
	c.Contract.ChangedInputRoles = []string{"source", "source", "manifest:appearance"}
	if err := c.CheckInputs(loaded); err == nil {
		t.Fatal("duplicate input declaration accepted")
	}
}
