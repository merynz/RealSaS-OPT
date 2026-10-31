package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/agentsession"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
	"github.com/merynz/RealSaS-OPT/platform/internal/campaign"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/externalstage"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

const maxArtifactReadBytes int64 = 64 << 20

// Local operator API. Commands use the transactional outbox; handlers never
// call a compiler script or rewrite a product revision themselves.
type API struct {
	CodeSHA string
	Pool    *pgxpool.Pool
	Graph   *stagegraph.Graph
	Store   artifactstore.Store
}

func respond(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func decode(w http.ResponseWriter, r *http.Request, value any) bool {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		respond(w, 400, map[string]string{"error": err.Error()})
		return false
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		respond(w, 400, map[string]string{"error": "exactly one JSON request is required"})
		return false
	}
	return true
}

func result(w http.ResponseWriter, value any, err error) {
	if err != nil {
		respond(w, 409, map[string]string{"error": err.Error()})
		return
	}
	respond(w, 202, value)
}

func (a API) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/live", func(w http.ResponseWriter, r *http.Request) { respond(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /v1/stages", func(w http.ResponseWriter, r *http.Request) { respond(w, 200, a.Graph.Stages()) })
	mux.HandleFunc("POST /v1/subjects", func(w http.ResponseWriter, r *http.Request) {
		var req input.SubjectRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := input.EnsureSubject(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/subjects/{id}/artifacts", func(w http.ResponseWriter, r *http.Request) {
		subjectID, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid subject id"})
			return
		}
		limit := 200
		if raw := r.URL.Query().Get("limit"); raw != "" {
			value, err := strconv.Atoi(raw)
			if err != nil || value <= 0 || value > 500 {
				respond(w, 400, map[string]string{"error": "artifact list limit must be between 1 and 500"})
				return
			}
			limit = value
		}
		out, err := registry.ListSubjectArtifacts(
			r.Context(),
			a.Pool,
			subjectID,
			r.URL.Query().Get("artifact_type"),
			r.URL.Query().Get("schema_version"),
			limit,
		)
		if errors.Is(err, registry.ErrSubjectNotFound) {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("PUT /v1/artifacts/bytes", func(w http.ResponseWriter, r *http.Request) {
		sha := r.URL.Query().Get("sha256")
		if err := artifactstore.ValidateSHA256(sha); err != nil {
			respond(w, 400, map[string]string{"error": err.Error()})
			return
		}
		if a.Store == nil {
			respond(w, 503, map[string]string{"error": "artifact store is required"})
			return
		}
		if r.ContentLength < 0 {
			respond(w, http.StatusLengthRequired, map[string]string{"error": "ARTIFACT_UPLOAD_CONTENT_LENGTH_REQUIRED"})
			return
		}
		if r.ContentLength > artifactstore.MaxArtifactObjectBytes {
			respond(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "ARTIFACT_UPLOAD_SIZE_LIMIT_EXCEEDED"})
			return
		}
		out, err := a.Store.PutStream(r.Context(), r.Body, sha, r.ContentLength)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/artifacts/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid artifact id"})
			return
		}
		out, err := registry.GetArtifact(r.Context(), a.Pool, id)
		if errors.Is(err, registry.ErrArtifactNotFound) {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("GET /v1/artifacts/{id}/bytes", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid artifact id"})
			return
		}
		record, err := registry.GetArtifact(r.Context(), a.Pool, id)
		if errors.Is(err, registry.ErrArtifactNotFound) {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		if err != nil {
			result(w, nil, err)
			return
		}
		if record.SizeBytes > maxArtifactReadBytes {
			respond(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "ARTIFACT_READ_SIZE_LIMIT_EXCEEDED"})
			return
		}
		if a.Store == nil {
			respond(w, 503, map[string]string{"error": "artifact store is required"})
			return
		}
		object := artifactstore.Object{ContentSHA256: record.ContentSHA256, StorageKey: record.StorageKey, SizeBytes: record.SizeBytes}
		if err := a.Store.Verify(r.Context(), object); err != nil {
			respond(w, 409, map[string]string{"error": "ARTIFACT_READ_VERIFY_FAILED:" + err.Error()})
			return
		}
		data, err := a.Store.GetBytes(r.Context(), record.StorageKey)
		if err != nil {
			respond(w, 409, map[string]string{"error": "ARTIFACT_READ_FAILED:" + err.Error()})
			return
		}
		if int64(len(data)) != record.SizeBytes || artifactstore.HashBytes(data) != record.ContentSHA256 {
			respond(w, 409, map[string]string{"error": "ARTIFACT_READ_POST_VERIFY_DRIFT"})
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", strconv.FormatInt(record.SizeBytes, 10))
		w.Header().Set("X-Content-SHA256", record.ContentSHA256)
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(data)
	})
	mux.HandleFunc("POST /v1/artifacts/import", func(w http.ResponseWriter, r *http.Request) {
		var req registry.ImportRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := registry.Import(r.Context(), a.Pool, a.Store, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/subject-inputs", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			SubjectID uuid.UUID       `json:"subject_id"`
			Bindings  []input.Binding `json:"bindings"`
			CreatedBy string          `json:"created_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		sort.Slice(req.Bindings, func(i, j int) bool { return req.Bindings[i].Role < req.Bindings[j].Role })
		out, err := input.Seal(r.Context(), a.Pool, req.SubjectID, req.Bindings, req.CreatedBy)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/releases", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Manifest  release.Manifest `json:"manifest"`
			CreatedBy string           `json:"created_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		out, err := release.Seal(r.Context(), a.Pool, a.Graph, req.Manifest, req.CreatedBy)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/research/attempts", func(w http.ResponseWriter, r *http.Request) {
		var req attempt.ResearchRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := attempt.StartResearch(r.Context(), a.Pool, a.Graph, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/research/attempts/{id}/stages/{stage}/external-evidence", func(w http.ResponseWriter, r *http.Request) {
		attemptID, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid attempt id"})
			return
		}
		var req struct {
			Bindings []externalstage.Binding `json:"bindings"`
			BoundBy  string                  `json:"bound_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		out, err := externalstage.Bind(r.Context(), a.Pool, externalstage.BindRequest{
			AttemptID: attemptID, StageID: r.PathValue("stage"), Bindings: req.Bindings, BoundBy: req.BoundBy,
		})
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/research/campaigns", func(w http.ResponseWriter, r *http.Request) {
		var req campaign.StartRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := campaign.Start(r.Context(), a.Pool, a.Graph, a.CodeSHA, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/research/campaigns/run", func(w http.ResponseWriter, r *http.Request) {
		var req campaign.RunRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := campaign.Run(r.Context(), a.Pool, a.Graph, a.CodeSHA, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/research/campaigns/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid campaign id"})
			return
		}
		out, err := campaign.Get(r.Context(), a.Pool, id)
		if err != nil {
			respond(w, 404, map[string]string{"error": err.Error()})
			return
		}
		respond(w, 200, out)
	})
	mux.HandleFunc("POST /v1/agents/enter", func(w http.ResponseWriter, r *http.Request) {
		var req agentsession.OpenRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := agentsession.Open(r.Context(), a.Pool, a.CodeSHA, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/agents/exit", func(w http.ResponseWriter, r *http.Request) {
		var req agentsession.CloseRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := agentsession.Close(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/agents/context/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "subject UUID required"})
			return
		}
		out, err := agentsession.Context(r.Context(), a.Pool, id, a.CodeSHA)
		result(w, out, err)
	})
	compile := func(research bool) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			var req command.CompileRequest
			if !decode(w, r, &req) {
				return
			}
			if research != (req.ResearchAttemptID != nil) {
				respond(w, 400, map[string]string{"error": "EXECUTION_LANE_REQUEST_MISMATCH"})
				return
			}
			if research && req.AgentSessionID == nil {
				respond(w, 409, map[string]string{"error": "RESEARCH_EXECUTION_REQUIRES_AGENT_SESSION"})
				return
			}
			req.DeployedCodeSHA = a.CodeSHA
			out, err := command.SubmitCompile(r.Context(), a.Pool, a.Graph, req)
			result(w, out, err)
		}
	}
	mux.HandleFunc("POST /v1/research/compile", compile(true))
	mux.HandleFunc("POST /v1/product/compile", compile(false))
	mux.HandleFunc("POST /v1/product/render", func(w http.ResponseWriter, r *http.Request) {
		var req command.RenderRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := command.SubmitRender(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/attempts/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid attempt id"})
			return
		}
		var state, kind string
		if err := a.Pool.QueryRow(r.Context(), "SELECT final_state,kind FROM attempts WHERE id=$1", id).Scan(&state, &kind); err != nil {
			respond(w, 404, map[string]string{"error": "attempt not found"})
			return
		}
		rows, err := a.Pool.Query(r.Context(), `SELECT event_type,payload,created_at FROM attempt_events WHERE attempt_id=$1 ORDER BY id`, id)
		if err != nil {
			result(w, nil, err)
			return
		}
		defer rows.Close()
		type eventRecord struct {
			Type      string          `json:"type"`
			Payload   json.RawMessage `json:"payload"`
			CreatedAt time.Time       `json:"created_at"`
		}
		events := make([]eventRecord, 0)
		for rows.Next() {
			var event eventRecord
			if err := rows.Scan(&event.Type, &event.Payload, &event.CreatedAt); err != nil {
				result(w, nil, err)
				return
			}
			events = append(events, event)
		}
		if err := rows.Err(); err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, map[string]any{"attempt_id": id, "kind": kind, "state": state, "events": events})
	})
	return mux
}
