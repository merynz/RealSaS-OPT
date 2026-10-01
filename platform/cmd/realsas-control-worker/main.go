package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/control"
	"github.com/merynz/RealSaS-OPT/platform/internal/dispatch"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		log.Fatal("REALSAS_DATABASE_URL is required")
	}
	artifactRoot := os.Getenv("REALSAS_ARTIFACT_ROOT")
	if artifactRoot == "" {
		log.Fatal("REALSAS_ARTIFACT_ROOT is required")
	}

	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()
	store, err := artifactstore.NewLocal(artifactRoot)
	if err != nil {
		log.Fatal(err)
	}

	temporalClient, err := client.Dial(client.Options{
		HostPort:  env("REALSAS_TEMPORAL_ADDRESS", client.DefaultHostPort),
		Namespace: env("REALSAS_TEMPORAL_NAMESPACE", "default"),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer temporalClient.Close()

	activities := control.Activities{Pool: pool, Store: store}
	w := worker.New(temporalClient, orchestration.ControlTaskQueue, worker.Options{})
	w.RegisterWorkflowWithOptions(
		orchestration.CapabilityWorkflow,
		workflow.RegisterOptions{Name: orchestration.CapabilityWorkflowName},
	)
	w.RegisterActivityWithOptions(
		activities.ResolveCapabilityGoal,
		activity.RegisterOptions{Name: orchestration.ResolveCapabilityGoalActivityName},
	)
	w.RegisterActivityWithOptions(
		activities.PrepareCapabilityExecution,
		activity.RegisterOptions{Name: orchestration.PrepareCapabilityExecutionActivityName},
	)
	w.RegisterActivityWithOptions(
		activities.CommitCapabilityResult,
		activity.RegisterOptions{Name: orchestration.CommitCapabilityResultActivityName},
	)
	w.RegisterActivityWithOptions(
		activities.RecordCapabilityActivityError,
		activity.RegisterOptions{Name: orchestration.RecordCapabilityActivityErrorActivityName},
	)
	w.RegisterActivityWithOptions(
		activities.FinalizeCapabilityGoal,
		activity.RegisterOptions{Name: orchestration.FinalizeCapabilityGoalActivityName},
	)

	if err := w.Start(); err != nil {
		log.Fatal(err)
	}
	defer w.Stop()

	publisher := dispatch.TemporalPublisher{Client: temporalClient}
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	slog.Info("realsas control worker started",
		"task_queue", orchestration.ControlTaskQueue,
		"surface", "RUN_CAPABILITY",
	)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			result, err := dispatch.DispatchOnce(ctx, pool, publisher, 50)
			if err != nil {
				slog.Error("outbox dispatch failed", "error", err)
				continue
			}
			if result.Claimed != 0 {
				slog.Info("outbox dispatched",
					"claimed", result.Claimed,
					"delivered", result.Delivered,
					"failed", result.Failed,
				)
			}
		}
	}
}
