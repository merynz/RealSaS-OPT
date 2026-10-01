package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func main() {
	planPath := flag.String("plan", "../canonical/MAINLINE_EXECUTION_PLAN_V2.json", "path to canonical 46-stage plan")
	stageID := flag.String("stage", "", "optional exact stage id")
	moduleID := flag.String("module", "", "optional exact module id")
	flag.Parse()

	data, err := os.ReadFile(*planPath)
	if err != nil {
		fatal(err)
	}
	graph, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		fatal(err)
	}
	registry, err := architecture.Build(graph)
	if err != nil {
		fatal(err)
	}
	if err := registry.Validate(); err != nil {
		fatal(err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	switch {
	case *stageID != "":
		value, ok := registry.Stage(*stageID)
		if !ok {
			fatal(fmt.Errorf("unknown stage %s", *stageID))
		}
		if err := enc.Encode(value); err != nil {
			fatal(err)
		}
	case *moduleID != "":
		value, ok := registry.Module(*moduleID)
		if !ok {
			fatal(fmt.Errorf("unknown module %s", *moduleID))
		}
		if err := enc.Encode(value); err != nil {
			fatal(err)
		}
	default:
		if err := enc.Encode(registry); err != nil {
			fatal(err)
		}
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
