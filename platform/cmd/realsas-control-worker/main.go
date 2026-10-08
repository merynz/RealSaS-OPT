package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/control"
	"github.com/merynz/RealSaS-OPT/platform/internal/dispatch"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func main() {
	reconcileOnce := flag.Bool("reconcile-once", false, "close only Attempts whose bound Temporal workflow failed terminally")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	databaseURL := requiredEnv("REALSAS_DATABASE_URL")
	temporalAddress := envOr("REALSAS_TEMPORAL_ADDRESS", "127.0.0.1:7233")
	temporalNamespace := envOr("REALSAS_TEMPORAL_NAMESPACE", "default")
	repoRoot := resolveRepoRoot()
	artifactRoot := envOr("REALSAS_ARTIFACT_ROOT", filepath.Join(userHome(), "realsas_platform_artifacts"))

	planBytes, err := os.ReadFile(filepath.Join(repoRoot, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	fatalIf(err, "read canonical execution plan")
	graph, err := stagegraph.ParseCanonicalPlan(planBytes)
	fatalIf(err, "parse canonical execution plan")
	pool, err := persistence.Open(ctx, databaseURL)
	fatalIf(err, "open postgres")
	defer pool.Close()
	store, err := artifactstore.NewLocal(artifactRoot)
	fatalIf(err, "open local artifact store")

	temporalClient, err := client.Dial(client.Options{
		HostPort:  temporalAddress,
		Namespace: temporalNamespace,
	})
	fatalIf(err, "connect temporal")
	defer temporalClient.Close()
	if *reconcileOnce {
		count, err := dispatch.ReconcileFailedAttempts(ctx, pool, temporalClient, 32)
		fatalIf(err, "reconcile failed workflows")
		slog.Info("terminal workflow reconciliation", "closed_attempts", count)
		return
	}

	activities := control.Activities{Pool: pool, Graph: graph, Store: store}
	fatalIf(activities.Validate(), "validate control activities")

	w := worker.New(temporalClient, orchestration.ControlTaskQueue, worker.Options{})
	w.RegisterWorkflowWithOptions(orchestration.CompileWorkflow, workflow.RegisterOptions{Name: orchestration.CompileWorkflowName})
	w.RegisterWorkflowWithOptions(orchestration.RenderWorkflow, workflow.RegisterOptions{Name: orchestration.RenderWorkflowName})
	w.RegisterWorkflowWithOptions(orchestration.CapabilityWorkflow, workflow.RegisterOptions{Name: orchestration.CapabilityWorkflowName})

	w.RegisterActivityWithOptions(activities.ResolveCompilePlan, activity.RegisterOptions{Name: orchestration.ResolveCompilePlanActivityName})
	w.RegisterActivityWithOptions(activities.PrepareStageExecution, activity.RegisterOptions{Name: orchestration.PrepareStageExecutionActivityName})
	w.RegisterActivityWithOptions(activities.BindReusedStage, activity.RegisterOptions{Name: orchestration.BindReusedStageActivityName})
	w.RegisterActivityWithOptions(activities.RecordStageActivityError, activity.RegisterOptions{Name: orchestration.RecordStageActivityErrorActivityName})
	w.RegisterActivityWithOptions(activities.CommitStageResult, activity.RegisterOptions{Name: orchestration.CommitStageResultActivityName})
	w.RegisterActivityWithOptions(activities.FinalizeCompile, activity.RegisterOptions{Name: orchestration.FinalizeCompileActivityName})

	w.RegisterActivityWithOptions(activities.ResolveRenderRequest, activity.RegisterOptions{Name: orchestration.ResolveRenderRequestActivityName})
	w.RegisterActivityWithOptions(activities.CommitRenderOutput, activity.RegisterOptions{Name: orchestration.CommitRenderOutputActivityName})

	w.RegisterActivityWithOptions(activities.ResolveCapabilityGoal, activity.RegisterOptions{Name: orchestration.ResolveCapabilityGoalActivityName})
	w.RegisterActivityWithOptions(activities.PrepareCapabilityExecution, activity.RegisterOptions{Name: orchestration.PrepareCapabilityExecutionActivityName})
	w.RegisterActivityWithOptions(activities.CommitCapabilityResult, activity.RegisterOptions{Name: orchestration.CommitCapabilityResultActivityName})
	w.RegisterActivityWithOptions(activities.RecordCapabilityActivityError, activity.RegisterOptions{Name: orchestration.RecordCapabilityActivityErrorActivityName})
	w.RegisterActivityWithOptions(activities.FinalizeCapabilityGoal, activity.RegisterOptions{Name: orchestration.FinalizeCapabilityGoalActivityName})

	fatalIf(w.Start(), "start temporal control worker")
	defer w.Stop()

	publisher := dispatch.TemporalPublisher{Client: temporalClient}
	go dispatchLoop(ctx, pool, publisher)
	go reconcileLoop(ctx, pool, temporalClient)

	slog.Info(
		"realsas-control-worker ready",
		"task_queue", orchestration.ControlTaskQueue,
		"temporal", temporalAddress,
		"namespace", temporalNamespace,
		"artifact_root", artifactRoot,
	)
	<-ctx.Done()
}

func reconcileLoop(ctx context.Context, pool *pgxpool.Pool, temporalClient client.Client) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		bounded, cancel := context.WithTimeout(ctx, 8*time.Second)
		count, err := dispatch.ReconcileFailedAttempts(bounded, pool, temporalClient, 32)
		cancel()
		if err != nil && ctx.Err() == nil {
			slog.Error("terminal workflow reconciliation failed", "error", err)
		} else if count > 0 {
			slog.Info("terminal workflow reconciliation", "closed_attempts", count)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func dispatchLoop(ctx context.Context, pool *pgxpool.Pool, publisher dispatch.TemporalPublisher) {
	ticker := time.NewTicker(dispatchInterval())
	defer ticker.Stop()
	for {
		result, err := dispatch.DispatchOnce(ctx, pool, publisher, dispatchBatchSize())
		if err != nil && !errors.Is(err, context.Canceled) {
			slog.Error("outbox dispatch failed", "error", err)
		} else if result.Delivered > 0 || result.Failed > 0 {
			slog.Info(
				"outbox dispatch",
				"claimed", result.Claimed,
				"delivered", result.Delivered,
				"failed", result.Failed,
			)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func requiredEnv(name string) string {
	value := os.Getenv(name)
	if value == "" {
		slog.Error("required environment variable missing", "name", name)
		os.Exit(2)
	}
	return value
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func userHome() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return os.TempDir()
	}
	return home
}

func resolveRepoRoot() string {
	if root := os.Getenv("REALSAS_REPO_ROOT"); root != "" {
		return root
	}
	cwd, err := os.Getwd()
	fatalIf(err, "resolve cwd")
	for _, candidate := range []string{cwd, filepath.Dir(cwd)} {
		if _, err := os.Stat(filepath.Join(candidate, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json")); err == nil {
			return candidate
		}
	}
	fatalIf(errors.New("canonical plan not found"), "resolve repository root")
	return ""
}

func fatalIf(err error, message string) {
	if err == nil {
		return
	}
	slog.Error(message, "error", err)
	os.Exit(1)
}

func dispatchBatchSize() int {
	raw := envOr("REALSAS_OUTBOX_BATCH_SIZE", "64")
	value, err := strconv.Atoi(raw)
	if err != nil || value < 1 {
		return 64
	}
	return value
}

func dispatchInterval() time.Duration {
	raw := envOr("REALSAS_OUTBOX_INTERVAL_MS", "500")
	value, err := strconv.Atoi(raw)
	if err != nil || value < 50 {
		value = 500
	}
	return time.Duration(value) * time.Millisecond
}
