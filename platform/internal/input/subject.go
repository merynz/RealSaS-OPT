package input

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type SubjectRequest struct {
	Slug        string `json:"slug"`
	DisplayName string `json:"display_name"`
	CreatedBy   string `json:"created_by"`
}

type Subject struct {
	SubjectID uuid.UUID `json:"subject_id"`
	Reused    bool      `json:"reused"`
}

func EnsureSubject(ctx context.Context, pool *pgxpool.Pool, req SubjectRequest) (Subject, error) {
	if req.Slug == "" || len(req.Slug) > 200 || strings.TrimSpace(req.DisplayName) == "" || len(req.DisplayName) > 300 || strings.TrimSpace(req.CreatedBy) == "" {
		return Subject{}, errors.New("valid slug, display_name and created_by are required")
	}
	for _, c := range req.Slug {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return Subject{}, errors.New("slug must contain lowercase letters, digits, hyphens or underscores")
		}
	}
	var out Subject
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		id := uuid.NewSHA1(uuid.NameSpaceURL, []byte("realsas:subject:"+req.Slug))
		tag, err := tx.Exec(ctx, `INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3) ON CONFLICT (slug) DO NOTHING`, id, req.Slug, req.DisplayName)
		if err != nil {
			return err
		}
		var actual uuid.UUID
		var name string
		var archived bool
		if err := tx.QueryRow(ctx, `SELECT id,display_name,archived_at IS NOT NULL FROM subjects WHERE slug=$1 FOR SHARE`, req.Slug).Scan(&actual, &name, &archived); err != nil {
			return err
		}
		if name != req.DisplayName || archived {
			return errors.New("SUBJECT_IDENTITY_CONFLICT_OR_ARCHIVED")
		}
		if tag.RowsAffected() > 0 {
			payload, _ := json.Marshal(map[string]any{"slug": req.Slug, "display_name": name})
			if _, err := tx.Exec(ctx, `INSERT INTO audit_events(actor,action,subject_id,payload) VALUES ($1,'SUBJECT_CREATED',$2,$3)`, req.CreatedBy, actual, payload); err != nil {
				return err
			}
		}
		out = Subject{SubjectID: actual, Reused: tag.RowsAffected() == 0}
		return nil
	})
	return out, err
}
