package product

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

var (
	ErrTargetNotFoundOrWrongSubject = errors.New("promotion target not found or subject mismatch")
	ErrAlreadyCurrent                = errors.New("promotion target already current")
	ErrRequiredRolesMissing          = errors.New("promotion required roles missing")
	ErrArtifactNotQualified          = errors.New("promotion artifact not qualified")
)

var requiredProductRoles = []string{
	"appearance",
	"geometry",
	"mechanical_mesh",
	"motion_library",
	"observation",
	"runtime_compatibility",
	"skeleton",
	"skin",
	"visual_presentation",
}

type PromotionRequest struct {
	SubjectID        uuid.UUID
	TargetRevisionID uuid.UUID
	RequestedBy      string
	Reason           string
}

type PromotionResult struct {
	PromotionID   uuid.UUID
	FromRevision  *uuid.UUID
	ToRevision    uuid.UUID
	NewLockVersion int64
}

type qualificationSnapshot struct {
	RequiredRoles []string `json:"required_roles"`
	BoundRoles    []string `json:"bound_roles"`
	AllQualified  bool     `json:"all_bound_artifacts_have_pass_qualification"`
}

func Promote(ctx context.Context, pool *pgxpool.Pool, req PromotionRequest) (PromotionResult, error) {
	if req.SubjectID == uuid.Nil || req.TargetRevisionID == uuid.Nil {
		return PromotionResult{}, errors.New("subject and target revision are required")
	}
	if req.RequestedBy == "" || req.Reason == "" {
		return PromotionResult{}, errors.New("requested_by and reason are required")
	}

	var result PromotionResult
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		// Lock the subject itself so first-time promotion is serialized too.
		var locked uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR UPDATE", req.SubjectID).Scan(&locked); err != nil {
			return fmt.Errorf("lock subject: %w", err)
		}

		var targetSubject uuid.UUID
		if err := tx.QueryRow(ctx,
			"SELECT subject_id FROM product_revisions WHERE id=$1",
			req.TargetRevisionID,
		).Scan(&targetSubject); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrTargetNotFoundOrWrongSubject
			}
			return err
		}
		if targetSubject != req.SubjectID {
			return ErrTargetNotFoundOrWrongSubject
		}

		var current uuid.UUID
		var lockVersion int64
		currentErr := tx.QueryRow(ctx,
			"SELECT product_revision_id, lock_version FROM subject_current_revision WHERE subject_id=$1 FOR UPDATE",
			req.SubjectID,
		).Scan(&current, &lockVersion)
		var from *uuid.UUID
		switch {
		case currentErr == nil:
			if current == req.TargetRevisionID {
				return ErrAlreadyCurrent
			}
			v := current
			from = &v
		case errors.Is(currentErr, pgx.ErrNoRows):
			lockVersion = 0
		default:
			return currentErr
		}

		rows, err := tx.Query(ctx, `
			SELECT pra.role,
			       EXISTS (
			         SELECT 1 FROM qualifications q
			         WHERE q.artifact_id = pra.artifact_id
			           AND q.result = 'PASS'
			       ) AS qualified
			FROM product_revision_artifacts pra
			WHERE pra.product_revision_id=$1
			ORDER BY pra.role
		`, req.TargetRevisionID)
		if err != nil {
			return err
		}
		defer rows.Close()

		bound := make(map[string]bool)
		var boundRoles []string
		for rows.Next() {
			var role string
			var qualified bool
			if err := rows.Scan(&role, &qualified); err != nil {
				return err
			}
			bound[role] = qualified
			boundRoles = append(boundRoles, role)
		}
		if err := rows.Err(); err != nil {
			return err
		}

		var missing []string
		for _, role := range requiredProductRoles {
			qualified, ok := bound[role]
			if !ok {
				missing = append(missing, role)
				continue
			}
			if !qualified {
				return fmt.Errorf("%w: %s", ErrArtifactNotQualified, role)
			}
		}
		if len(missing) != 0 {
			return fmt.Errorf("%w: %v", ErrRequiredRolesMissing, missing)
		}
		sort.Strings(boundRoles)
		snapshot, err := json.Marshal(qualificationSnapshot{
			RequiredRoles: append([]string(nil), requiredProductRoles...),
			BoundRoles:    boundRoles,
			AllQualified:  true,
		})
		if err != nil {
			return err
		}

		promotionID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO promotions
			  (id, subject_id, from_revision_id, to_revision_id, reason, requested_by, qualification_snapshot)
			VALUES ($1,$2,$3,$4,$5,$6,$7)
		`, promotionID, req.SubjectID, from, req.TargetRevisionID, req.Reason, req.RequestedBy, snapshot); err != nil {
			return err
		}

		newLock := lockVersion + 1
		if from == nil {
			if _, err := tx.Exec(ctx, `
				INSERT INTO subject_current_revision(subject_id, product_revision_id, lock_version)
				VALUES ($1,$2,$3)
			`, req.SubjectID, req.TargetRevisionID, newLock); err != nil {
				return err
			}
		} else {
			if _, err := tx.Exec(ctx, `
				UPDATE subject_current_revision
				SET product_revision_id=$2, lock_version=$3, updated_at=now()
				WHERE subject_id=$1
			`, req.SubjectID, req.TargetRevisionID, newLock); err != nil {
				return err
			}
		}

		auditPayload, _ := json.Marshal(map[string]any{
			"promotion_id":      promotionID.String(),
			"from_revision_id":  from,
			"to_revision_id":    req.TargetRevisionID.String(),
			"reason":            req.Reason,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor, action, subject_id, product_revision_id, payload)
			VALUES ($1,'PRODUCT_REVISION_PROMOTED',$2,$3,$4)
		`, req.RequestedBy, req.SubjectID, req.TargetRevisionID, auditPayload); err != nil {
			return err
		}

		outboxPayload, _ := json.Marshal(map[string]any{
			"promotion_id":       promotionID.String(),
			"product_revision_id": req.TargetRevisionID.String(),
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO outbox_events(aggregate_type, aggregate_id, event_type, payload)
			VALUES ('Subject',$1,'ProductRevisionPromoted',$2)
		`, req.SubjectID, outboxPayload); err != nil {
			return err
		}

		result = PromotionResult{
			PromotionID:   promotionID,
			FromRevision:  from,
			ToRevision:    req.TargetRevisionID,
			NewLockVersion: newLock,
		}
		return nil
	})
	return result, err
}
