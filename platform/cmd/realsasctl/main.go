package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/merynz/RealSaS-OPT/platform/internal/agentsession"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
)

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("command required: stages, status, experiments, experiment, experiment-create, experiment-transition, experiment-link, models, model, model-channels, model-release, model-evaluate, model-promote, subject, artifact-put, artifact-import, input-seal, attempt, campaign, release, agent-context, agent-enter, agent-exit, research-start, research-run, capability-run, campaign-start, campaign-run, product-compile, product-render")
	}
	routes := map[string]string{
		"subject":               "/v1/subjects",
		"artifact-import":       "/v1/artifacts/import",
		"input-seal":            "/v1/subject-inputs",
		"release":               "/v1/releases",
		"research-start":        "/v1/research/attempts",
		"research-run":          "/v1/research/compile",
		"capability-run":        "/v1/capabilities/run",
		"campaign-start":        "/v1/research/campaigns",
		"campaign-run":          "/v1/research/campaigns/run",
		"product-compile":       "/v1/product/compile",
		"product-render":        "/v1/product/render",
		"experiment-create":     "/v1/experiments",
		"experiment-transition": "/v1/experiments/transition",
		"experiment-link":       "/v1/experiments/link",
		"model-release":         "/v1/model-releases",
		"model-evaluate":        "/v1/model-releases/evaluations",
		"model-promote":         "/v1/model-releases/promote",
	}
	routes["agent-enter"], routes["agent-exit"] = "/v1/agents/enter", "/v1/agents/exit"
	flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
	endpoint := flags.String("api", "http://127.0.0.1:8080", "operator API")
	requestFile := flags.String("request", "-", "JSON request file, or - for stdin")
	id := flags.String("id", "", "attempt, campaign, experiment or model release UUID")
	repo := flags.String("repo", ".", "canonical source checkout for execution commands")
	file := flags.String("file", "", "exact artifact bytes to upload")
	sha := flags.String("sha256", "", "expected artifact content SHA256")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	method := http.MethodPost
	path, ok := routes[args[0]]
	var body []byte
	var artifactPath string
	var artifactSize int64
	if args[0] == "artifact-put" {
		if *file == "" || len(*sha) != 64 {
			return fmt.Errorf("artifact-put requires --file and --sha256")
		}
		if err := artifactstore.ValidateSHA256(*sha); err != nil {
			return err
		}
		info, err := os.Stat(*file)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("artifact upload requires a regular file")
		}
		if err := artifactstore.ValidateObjectIdentity(*sha, info.Size()); err != nil {
			return err
		}
		actualSHA, err := hashFile(*file)
		if err != nil {
			return err
		}
		if actualSHA != *sha {
			return fmt.Errorf("ARTIFACT_UPLOAD_HASH_MISMATCH")
		}
		method = http.MethodPut
		path = "/v1/artifacts/bytes?sha256=" + *sha
		ok = true
		artifactPath = *file
		artifactSize = info.Size()
	}
	switch args[0] {
	case "stages":
		method, path, ok = http.MethodGet, "/v1/stages", true
	case "status":
		method, path, ok = http.MethodGet, "/v1/ops/status", true
	case "experiments":
		method, path, ok = http.MethodGet, "/v1/experiments", true
	case "experiment":
		method, path, ok = http.MethodGet, "/v1/experiments/"+*id, *id != ""
	case "models":
		method, path, ok = http.MethodGet, "/v1/model-releases", true
	case "model":
		method, path, ok = http.MethodGet, "/v1/model-releases/"+*id, *id != ""
	case "model-channels":
		method, path, ok = http.MethodGet, "/v1/model-channels", true
	case "attempt":
		method, path, ok = http.MethodGet, "/v1/attempts/"+*id, *id != ""
	case "campaign":
		method, path, ok = http.MethodGet, "/v1/research/campaigns/"+*id, *id != ""
	case "agent-context":
		method, path, ok = http.MethodGet, "/v1/agents/context/"+*id, *id != ""
	}
	if !ok {
		return fmt.Errorf("unknown command or missing --id: %s", args[0])
	}
	if method == http.MethodPost {
		var err error
		if *requestFile == "-" {
			body, err = io.ReadAll(io.LimitReader(os.Stdin, 8<<20))
		} else {
			body, err = os.ReadFile(*requestFile)
		}
		if err != nil {
			return err
		}
	}
	switch args[0] {
	case "agent-enter", "release", "research-start", "research-run", "capability-run", "campaign-start", "campaign-run", "product-compile", "product-render":
		codeSHA, err := agentsession.CanonicalCheckout(context.Background(), *repo)
		if err != nil {
			return err
		}
		if args[0] == "agent-enter" {
			var entry agentsession.OpenRequest
			decoder := json.NewDecoder(bytes.NewReader(body))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&entry); err != nil {
				return err
			}
			if err := decoder.Decode(&struct{}{}); err != io.EOF {
				return fmt.Errorf("exactly one entry JSON required")
			}
			if err := agentsession.ValidateCode(entry.CanonicalCodeSHA, codeSHA); err != nil {
				return err
			}
		}
	}

	var requestBody io.Reader = bytes.NewReader(body)
	var uploadFile *os.File
	if artifactPath != "" {
		var err error
		uploadFile, err = os.Open(artifactPath)
		if err != nil {
			return err
		}
		defer uploadFile.Close()
		requestBody = uploadFile
	}
	req, err := http.NewRequest(method, strings.TrimRight(*endpoint, "/")+path, requestBody)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if method == http.MethodPut {
		req.Header.Set("Content-Type", "application/octet-stream")
		req.ContentLength = artifactSize
	}
	clientTimeout := 30 * time.Second
	if args[0] == "artifact-put" {
		clientTimeout = 30 * time.Minute
	}
	response, err := (&http.Client{Timeout: clientTimeout}).Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 16<<20))
	if err != nil {
		return err
	}
	if response.StatusCode >= 400 {
		return fmt.Errorf("%s: %s", response.Status, strings.TrimSpace(string(data)))
	}
	_, err = os.Stdout.Write(data)
	return err
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
