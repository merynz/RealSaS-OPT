package httpapi

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/attempt"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

// Local operator API. Commands use the transactional outbox; handlers never
// call a compiler script or rewrite a product revision themselves.
type API struct {
	Pool  *pgxpool.Pool
	Graph *stagegraph.Graph
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
			// Preserve the stored diagnostic payload without deriving authority.
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
