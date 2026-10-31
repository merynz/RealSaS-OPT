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
	CodeSHA            string
	Pool               *pgxpool.Pool
	Graph              *stagegraph.Graph
	Store              artifactstore.Store
	ProductGatewayToken string
	OperatorToken       string
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
	respond(w, 200, value)
}

func (a API) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/commands/compile", func(w http.ResponseWriter, r *http.Request) {
		var req command.CompileRequest
		if !decode(w, r, &req) {
			return
		}
		if req.EngineReleaseID == uuid.Nil || req.SubjectInputID == uuid.Nil || req.TargetStageID == "" || req.IdempotencyKey == "" || req.RequestedBy == "" {
			respond(w, 400, map[string]string{"error": "engine_release_id, subject_input_id, target_stage_id, idempotency_key and requested_by are required"})
			return
		}
		if err := release.RequireCanonicalMain(r.Context(), a.Pool, req.EngineReleaseID, a.CodeSHA); err != nil {
			result(w, nil, err)
			return
		}
		out, err := command.SubmitCompile(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/commands/render", func(w http.ResponseWriter, r *http.Request) {
		var req command.RenderRequest
		if !decode(w, r, &req) {
			return
		}
		if req.ProductRevisionID == uuid.Nil || req.RenderRequestID == uuid.Nil || req.RenderRequestSemanticSHA256 == "" || req.IdempotencyKey == "" || req.RequestedBy == "" {
			respond(w, 400, map[string]string{"error": "product_revision_id, render_request_id, render_request_semantic_sha256, idempotency_key and requested_by are required"})
			return
		}
		out, err := command.SubmitRender(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/attempts/{attempt}/external-stage-ready", func(w http.ResponseWriter, r *http.Request) {
		attemptID, err := uuid.Parse(r.PathValue("attempt"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid attempt id"})
			return
		}
		var req command.ExternalStageReadyRequest
		if !decode(w, r, &req) {
			return
		}
		req.AttemptID = attemptID
		if req.WorkflowID == "" || req.StageID == "" || req.EvidenceSHA256 == "" || req.IdempotencyKey == "" || req.RequestedBy == "" {
			respond(w, 400, map[string]string{"error": "workflow_id, stage_id, evidence_sha256, idempotency_key and requested_by are required"})
			return
		}
		out, err := command.SubmitExternalStageReady(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/attempts/{attempt}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("attempt"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid attempt id"})
			return
		}
		out, err := attempt.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/attempts/{attempt}/events", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("attempt"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid attempt id"})
			return
		}
		limit := 200
		if raw := r.URL.Query().Get("limit"); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 || parsed > 1000 {
				respond(w, 400, map[string]string{"error": "limit must be 1..1000"})
				return
			}
			limit = parsed
		}
		out, err := attempt.Events(r.Context(), a.Pool, id, limit)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/artifacts/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid artifact id"})
			return
		}
		out, err := registry.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/artifacts/{id}/bytes", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("id"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid artifact id"})
			return
		}
		item, err := registry.Read(r.Context(), a.Pool, id)
		if err != nil {
			result(w, nil, err)
			return
		}
		if item.SizeBytes > maxArtifactReadBytes {
			respond(w, 413, map[string]string{"error": "artifact exceeds direct read limit"})
			return
		}
		payload, err := a.Store.GetBytes(r.Context(), item.StorageKey)
		if err != nil {
			result(w, nil, err)
			return
		}
		if int64(len(payload)) != item.SizeBytes {
			respond(w, 409, map[string]string{"error": "artifact byte length drift"})
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-Content-SHA256", item.ContentSHA256)
		w.WriteHeader(200)
		_, _ = w.Write(payload)
	})
	mux.HandleFunc("POST /v1/artifacts/bytes", func(w http.ResponseWriter, r *http.Request) {
		declaredLength, err := strconv.ParseInt(r.Header.Get("X-Content-Length"), 10, 64)
		if err != nil || declaredLength < 0 {
			respond(w, 400, map[string]string{"error": "valid X-Content-Length required"})
			return
		}
		if declaredLength > artifactstore.MaxArtifactObjectBytes {
			respond(w, 413, map[string]string{"error": "artifact exceeds storage limit"})
			return
		}
		reader := http.MaxBytesReader(w, r.Body, declaredLength+1)
		out, err := a.Store.PutStream(r.Context(), reader, r.Header.Get("X-Content-SHA256"), declaredLength)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/artifacts/import", func(w http.ResponseWriter, r *http.Request) {
		var req registry.ImportRequest
		if !decode(w, r, &req) {
			return
		}
		if req.ArtifactType == "" || req.SchemaVersion == "" || req.Object.ContentSHA256 == "" || req.Object.StorageKey == "" || req.CreatedBy == "" {
			respond(w, 400, map[string]string{"error": "artifact_type, schema_version, object and created_by are required"})
			return
		}
		out, err := registry.Import(r.Context(), a.Pool, a.Store, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/subjects", func(w http.ResponseWriter, r *http.Request) {
		var req input.SubjectRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := input.EnsureSubject(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/subject-inputs", func(w http.ResponseWriter, r *http.Request) {
		var req input.CreateRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := input.Create(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/commands/{command}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("command"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid command id"})
			return
		}
		out, err := command.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/engine-releases", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Manifest release.Manifest `json:"manifest"`
			CreatedBy string           `json:"created_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		if req.CreatedBy == "" {
			respond(w, 400, map[string]string{"error": "created_by required"})
			return
		}
		out, err := release.Seal(r.Context(), a.Pool, a.Graph, req.Manifest, req.CreatedBy)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/engine-releases/{release}/source-authority", func(w http.ResponseWriter, r *http.Request) {
		releaseID, err := uuid.Parse(r.PathValue("release"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid release id"})
			return
		}
		var req struct {
			Authority string `json:"authority"`
			CodeSHA   string `json:"code_sha"`
			BoundBy   string `json:"bound_by"`
		}
		if !decode(w, r, &req) {
			return
		}
		if req.Authority != "CANONICAL_MAIN" {
			respond(w, 400, map[string]string{"error": "only CANONICAL_MAIN authority is accepted"})
			return
		}
		out, err := release.BindCanonicalMain(r.Context(), a.Pool, releaseID, req.CodeSHA, req.BoundBy)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/engine-releases/{release}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("release"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid release id"})
			return
		}
		out, err := release.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/engine-releases/{release}/source-authority", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("release"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid release id"})
			return
		}
		out, err := release.ReadSourceBinding(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/engine-releases", func(w http.ResponseWriter, r *http.Request) {
		limit := 100
		if raw := r.URL.Query().Get("limit"); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 || parsed > 500 {
				respond(w, 400, map[string]string{"error": "limit must be 1..500"})
				return
			}
			limit = parsed
		}
		out, err := release.List(r.Context(), a.Pool, limit)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/research-campaigns", func(w http.ResponseWriter, r *http.Request) {
		var req campaign.CreateRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := campaign.Create(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/agent-sessions", func(w http.ResponseWriter, r *http.Request) {
		var req agentsession.CreateRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := agentsession.Create(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/agent-sessions/{session}/commands", func(w http.ResponseWriter, r *http.Request) {
		sessionID, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid session id"})
			return
		}
		var req agentsession.AppendCommandRequest
		if !decode(w, r, &req) {
			return
		}
		req.SessionID = sessionID
		out, err := agentsession.AppendCommand(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/external-stage/sessions", func(w http.ResponseWriter, r *http.Request) {
		var req externalstage.CreateSessionRequest
		if !decode(w, r, &req) {
			return
		}
		out, err := externalstage.CreateSession(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("POST /v1/external-stage/sessions/{session}/evidence", func(w http.ResponseWriter, r *http.Request) {
		sessionID, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid external stage session id"})
			return
		}
		var req externalstage.RecordEvidenceRequest
		if !decode(w, r, &req) {
			return
		}
		req.SessionID = sessionID
		out, err := externalstage.RecordEvidence(r.Context(), a.Pool, req)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/research-campaigns", func(w http.ResponseWriter, r *http.Request) {
		limit := 100
		if raw := r.URL.Query().Get("limit"); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 || parsed > 500 {
				respond(w, 400, map[string]string{"error": "limit must be 1..500"})
				return
			}
			limit = parsed
		}
		out, err := campaign.List(r.Context(), a.Pool, limit)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/research-campaigns/{campaign}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("campaign"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid campaign id"})
			return
		}
		out, err := campaign.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/research-campaigns/{campaign}/open-attempts", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("campaign"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid campaign id"})
			return
		}
		out, err := campaign.OpenAttempts(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/agent-sessions/{session}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid agent session id"})
			return
		}
		out, err := agentsession.Read(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/agent-sessions/{session}/events", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid agent session id"})
			return
		}
		limit := 200
		if raw := r.URL.Query().Get("limit"); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 || parsed > 1000 {
				respond(w, 400, map[string]string{"error": "limit must be 1..1000"})
				return
			}
			limit = parsed
		}
		out, err := agentsession.Events(r.Context(), a.Pool, id, limit)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/external-stage/sessions/{session}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid external stage session id"})
			return
		}
		out, err := externalstage.ReadSession(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	mux.HandleFunc("GET /v1/external-stage/sessions/{session}/evidence", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("session"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid external stage session id"})
			return
		}
		out, err := externalstage.ListEvidence(r.Context(), a.Pool, id)
		result(w, out, err)
	})
	return mux
}

func (a API) HandlerWithSubjectIdentityRead() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/subjects/{subject}", func(w http.ResponseWriter, r *http.Request) {
		id, err := uuid.Parse(r.PathValue("subject"))
		if err != nil {
			respond(w, 400, map[string]string{"error": "invalid subject id"})
			return
		}
		var slug, display string
		var createdAt time.Time
		if err := a.Pool.QueryRow(r.Context(), `
			SELECT slug,display_name,created_at FROM subjects WHERE id=$1
		`, id).Scan(&slug, &display, &createdAt); err != nil {
			result(w, nil, err)
			return
		}
		respond(w, 200, map[string]any{"id": id, "slug": slug, "display_name": display, "created_at": createdAt})
	})
	mux.Handle("/", a.Handler())
	return mux
}
