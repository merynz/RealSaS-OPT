package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type environment struct {
	root  string
	graph *stagegraph.Graph
}

func repoRoot() (string, error) {
	if raw := os.Getenv("REALSAS_REPO_ROOT"); raw != "" {
		return filepath.Abs(raw)
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for cur := wd; ; cur = filepath.Dir(cur) {
		if _, err := os.Stat(filepath.Join(cur, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json")); err == nil {
			return cur, nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return "", errors.New("repository root not found")
		}
	}
}

func loadEnvironment() (environment, error) {
	root, err := repoRoot()
	if err != nil {
		return environment{}, err
	}
	raw, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		return environment{}, err
	}
	graph, err := stagegraph.ParseCanonicalPlan(raw)
	if err != nil {
		return environment{}, err
	}
	return environment{root: root, graph: graph}, nil
}

func fileSHA(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func adapterFile(root, adapter string) (string, error) {
	module, _, ok := strings.Cut(adapter, ":")
	if !ok || module == "" {
		return "", fmt.Errorf("invalid adapter ref %q", adapter)
	}
	path := filepath.Join(root, filepath.FromSlash(strings.ReplaceAll(module, ".", "/")+".py"))
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("adapter %s: %w", adapter, err)
	}
	return path, nil
}

func openDB(ctx context.Context) (*persistenceHandle, error) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		return nil, errors.New("REALSAS_DATABASE_URL is required")
	}
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		return nil, err
	}
	return &persistenceHandle{pool: pool}, nil
}

type persistenceHandle struct {
	pool interface {
		Close()
	}
}

func bootstrapRuntimeRelease(ctx context.Context, env environment, createdBy, name string) error {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		return errors.New("REALSAS_DATABASE_URL is required")
	}
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()

	bindings := make([]release.StageBinding, 0, env.graph.StageCount())
	for _, stage := range env.graph.Stages() {
		path, err := adapterFile(env.root, stage.Adapter)
		if err != nil {
			return err
		}
		impl, err := fileSHA(path)
		if err != nil {
			return err
		}
		policy, err := semantic.JSONSHA256(stage.Policy)
		if err != nil {
			return err
		}
		bindings = append(bindings, release.StageBinding{
			Ordinal:              stage.Ordinal,
			StageID:              stage.ID,
			ImplementationSHA256: impl,
			PolicySHA256:         policy,
			SemanticParameters: map[string]any{
				"adapter_ref": stage.Adapter,
				"group":       stage.Group,
			},
		})
	}
	sealed, err := release.Seal(ctx, pool, env.graph, release.Manifest{
		Name:    name,
		Purpose: release.PurposeResearch,
		Stages:  bindings,
	}, createdBy)
	if err != nil {
		return err
	}

	arch, err := architecture.Build(env.graph)
	if err != nil {
		return err
	}
	registry, err := capability.Build(ctx, capability.RuntimeProvider{})
	if err != nil {
		return err
	}
	_ = arch // architecture validation above intentionally proves current graph ownership.

	workerPath := filepath.Join(env.root, "compiler", "realsas_compiler_services", "platform_worker_v1.py")
	workerSHA, err := fileSHA(workerPath)
	if err != nil {
		return err
	}
	policySHA, err := semantic.JSONSHA256(map[string]any{
		"operation":                    "RENDER_COMPILER_RUN",
		"render_only":                   true,
		"fit_train_calibration":         false,
		"product_promotion":             false,
		"source":                        "SEALED_COMPILER_RUN",
		"runtime_package_stage":         "43_RSS_MATERIALIZE_COMPACT",
	})
	if err != nil {
		return err
	}
	paramsSHA, err := semantic.JSONSHA256(map[string]any{
		"contract": "RealSaS.DeveloperCompilerRunRenderParameters.v1",
	})
	if err != nil {
		return err
	}
	var snapshotBindings []capability.ReleaseBinding
	for _, descriptor := range registry.Descriptors() {
		snapshotBindings = append(snapshotBindings, capability.ReleaseBinding{
			Descriptor: descriptor,
			Version: capability.VersionIdentity{
				ImplementationSHA256: workerSHA,
				PolicySHA256:         policySHA,
				ParametersSHA256:     paramsSHA,
			},
		})
	}
	setSHA, err := capability.SealReleaseSnapshot(ctx, pool, sealed.ReleaseID, snapshotBindings, createdBy)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"schema":                "RealSaS.ForgeBootstrapRuntimeRelease.v1",
		"release_id":            sealed.ReleaseID.String(),
		"release_sha256":        sealed.ReleaseSHA256,
		"capability_set_sha256": setSHA,
		"reused":                sealed.Reused,
	})
}

func submitRender(ctx context.Context, compilerRunID, clipID, viewID, createdBy, key, releaseRaw string, fps float64) error {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		return errors.New("REALSAS_DATABASE_URL is required")
	}
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()
	releaseID, err := uuid.Parse(releaseRaw)
	if err != nil {
		return err
	}
	params := map[string]any{
		"compiler_run_id": compilerRunID,
		"fps":             fps,
	}
	if clipID != "" {
		params["clip_id"] = clipID
	}
	if viewID != "" {
		params["view_id"] = viewID
	}
	receipt, err := command.SubmitCapabilityRun(ctx, pool, command.CapabilityRunRequest{
		EngineReleaseID: releaseID,
		Targets:         []string{capability.RuntimeRenderCompilerRunCapabilityID},
		Parameters:      params,
		IdempotencyKey:  key,
		RequestedBy:     createdBy,
	})
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"schema":            "RealSaS.ForgeCapabilityReceipt.v1",
		"command_id":        receipt.CommandID.String(),
		"attempt_id":        receipt.AttemptID.String(),
		"execution_goal_id": receipt.ExecutionGoalID.String(),
		"reused":            receipt.ReusedIdempotencyKey,
	})
}

func waitAttempt(ctx context.Context, attemptRaw string, timeout time.Duration) error {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		return errors.New("REALSAS_DATABASE_URL is required")
	}
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()
	attemptID, err := uuid.Parse(attemptRaw)
	if err != nil {
		return err
	}
	deadline := time.Now().Add(timeout)
	for {
		var state string
		if err := pool.QueryRow(ctx, "SELECT final_state FROM attempts WHERE id=$1", attemptID).Scan(&state); err != nil {
			return err
		}
		if state != "OPEN" {
			rows, err := pool.Query(ctx, `
				SELECT aa.role,t.name,t.schema_version,a.semantic_sha256,a.content_sha256,a.storage_key,a.size_bytes
				FROM attempt_artifacts aa
				JOIN artifacts a ON a.id=aa.artifact_id
				JOIN artifact_types t ON t.id=a.artifact_type_id
				WHERE aa.attempt_id=$1
				ORDER BY aa.role
			`, attemptID)
			if err != nil {
				return err
			}
			defer rows.Close()
			var artifacts []map[string]any
			for rows.Next() {
				var role, typ, schema, semanticSHA, contentSHA, storageKey string
				var size int64
				if err := rows.Scan(&role, &typ, &schema, &semanticSHA, &contentSHA, &storageKey, &size); err != nil {
					return err
				}
				artifacts = append(artifacts, map[string]any{
					"role": role, "artifact_type": typ, "schema_version": schema,
					"semantic_sha256": semanticSHA, "content_sha256": contentSHA,
					"storage_key": storageKey, "size_bytes": size,
				})
			}
			if err := rows.Err(); err != nil {
				return err
			}
			if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
				"schema":     "RealSaS.ForgeAttemptStatus.v1",
				"attempt_id": attemptID.String(),
				"state":      state,
				"artifacts":  artifacts,
			}); err != nil {
				return err
			}
			if state != "PASS" {
				return fmt.Errorf("attempt finished with state %s", state)
			}
			return nil
		}
		if time.Now().After(deadline) {
			return errors.New("attempt wait timeout")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: realsas-forge <bootstrap-runtime-release|render-run|wait-attempt> [flags]")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	ctx := context.Background()
	env, err := loadEnvironment()
	if err != nil {
		panic(err)
	}
	switch os.Args[1] {
	case "bootstrap-runtime-release":
		fs := flag.NewFlagSet("bootstrap-runtime-release", flag.ExitOnError)
		name := fs.String("name", "forge-runtime-"+time.Now().UTC().Format("20060102"), "research release name")
		actor := fs.String("actor", "forge", "audit actor")
		_ = fs.Parse(os.Args[2:])
		err = bootstrapRuntimeRelease(ctx, env, *actor, *name)
	case "render-run":
		fs := flag.NewFlagSet("render-run", flag.ExitOnError)
		releaseID := fs.String("release-id", "", "sealed research EngineRelease UUID")
		runID := fs.String("compiler-run-id", "", "sealed compiler run id")
		clip := fs.String("clip", "", "clip id or unique substring")
		view := fs.String("view", "", "view id or unique substring")
		fps := fs.Float64("fps", 0, "GIF fps; 0 derives from clip")
		key := fs.String("idempotency-key", "", "command idempotency key")
		actor := fs.String("actor", "forge", "audit actor")
		_ = fs.Parse(os.Args[2:])
		if *releaseID == "" || *runID == "" || *key == "" {
			err = errors.New("release-id, compiler-run-id and idempotency-key are required")
			break
		}
		err = submitRender(ctx, *runID, *clip, *view, *actor, *key, *releaseID, *fps)
	case "wait-attempt":
		fs := flag.NewFlagSet("wait-attempt", flag.ExitOnError)
		attemptID := fs.String("attempt-id", "", "Attempt UUID")
		timeout := fs.Duration("timeout", 20*time.Minute, "wait timeout")
		_ = fs.Parse(os.Args[2:])
		if *attemptID == "" {
			err = errors.New("attempt-id is required")
			break
		}
		err = waitAttempt(ctx, *attemptID, *timeout)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
