package main

import (
	"context"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/httpapi"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func main() {
	root := os.Getenv("REALSAS_REPO_ROOT")
	if root == "" {
		root = ".."
	}
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		slog.Error("read plan", "error", err)
		os.Exit(1)
	}
	graph, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		slog.Error("parse plan", "error", err)
		os.Exit(1)
	}
	pool, err := persistence.Open(context.Background(), os.Getenv("REALSAS_DATABASE_URL"))
	if err != nil {
		slog.Error("open database", "error", err)
		os.Exit(1)
	}
	defer pool.Close()
	artifactRoot := os.Getenv("REALSAS_ARTIFACT_ROOT")
	if artifactRoot == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			slog.Error("home directory", "error", err)
			os.Exit(1)
		}
		artifactRoot = filepath.Join(home, "realsas_platform_artifacts")
	}
	store, err := artifactstore.NewLocal(artifactRoot)
	if err != nil {
		slog.Error("artifact store", "error", err)
		os.Exit(1)
	}
	handler := (httpapi.API{Pool: pool, Graph: graph, Store: store}).Handler()
	server := &http.Server{
		Addr:              "127.0.0.1:8080",
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	slog.Info("realsas-api starting", "addr", server.Addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("realsas-api failed", "error", err)
		os.Exit(1)
	}
}
