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
	ErrAlreadyCurrent               = errors.New("promotion target already current")
	ErrProductContractRequired      = errors.New("promotion target requires sealed product contract")
	ErrRequiredRolesMissing         = errors.New("promotion required roles missing")
	ErrArtifactNotQualified         = errors.New("promotion artifact not qualified")
	ErrArtifactContractMismatch     = errors.New("promotion artifact violates product contract")
)

type PromotionRequest struct {
	SubjectID        uuid.UUID
	TargetRevisionID uuid.UUID
	RequestedBy      string
	Reason           string
}

type PromotionResult struct {
	PromotionID    uuid.UUID
	FromRevision   *uuid.UUID
	ToRevision     uuid.UUID
	NewLockVersion int64
}

type qualificationSnapshot struct {
	ProductContractID     string   `json:"product_contract_id"`
	ProductContractSHA256 string   `json:"product_contract_sha256"`
	RequiredRoles         []string `json:"required_roles"`
	BoundRoles            []string `json:"bound_roles"`
	AllQualified          bool     `json:"all_bound_artifacts_have_required_qualification"`
}

type boundArtifact struct {
	ID            uuid.UUID
	ArtifactType  string
	SchemaVersion string
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
		var locked uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM subjects WHERE id=$1 FOR UPDATE", req.SubjectID).Scan(&locked); err != nil {
			return fmt.Errorf("lock subject: %w", err)
		}

		var targetSubject uuid.UUID
		var productContractID *uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,product_contract_id
			FROM product_revisions
			WHERE id=$1
		`, req.TargetRevisionID).Scan(&targetSubject, &productContractID); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrTargetNotFoundOrWrongSubject
			}
			return err
		}
		if targetSubject != req.SubjectID {
			return ErrTargetNotFoundOrWrongSubject
		}
		if productContractID == nil {
			return ErrProductContractRequired
		}
		contract, contractSHA, err := loadContractTx(ctx, tx, *productContractID)
		if err != nil {
			return err
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
			SELECT pra.role,pra.artifact_id,t.name,t.schema_version
			FROM product_revision_artifacts pra
			JOIN artifacts a ON a.id=pra.artifact_id
			JOIN artifact_types t ON t.id=a.artifact_type_id
			WHERE pra.product_revision_id=$1
			ORDER BY pra.role
		`, req.TargetRevisionID)
		if err != nil {
			return err
		}
		defer rows.Close()
		bound := map[string]boundArtifact{}
		var boundRoles []string
		for rows.Next() {
			var role string
			var artifact boundArtifact
			if err := rows.Scan(&role, &artifact.ID, &artifact.ArtifactType, &artifact.SchemaVersion); err != nil {
				return err
			}
			bound[role] = artifact
			boundRoles = append(boundRoles, role)
		}
		if err := rows.Err(); err != nil {
			return err
		}

		var requiredRoles []string
		var missing []string
		for _, rule := range contract.Roles {
			if !rule.Required {
				continue
			}
			requiredRoles = append(requiredRoles, rule.Role)
			artifact, ok := bound[rule.Role]
			if !ok {
				missing = append(missing, rule.Role)
				continue
			}
			if rule.ArtifactType != "" && artifact.ArtifactType != rule.ArtifactType {
				return fmt.Errorf("%w: role=%s type=%s want=%s", ErrArtifactContractMismatch, rule.Role, artifact.ArtifactType, rule.ArtifactType)
			}
			if rule.SchemaVersion != "" && artifact.SchemaVersion != rule.SchemaVersion {
				return fmt.Errorf("%w: role=%s schema=%s want=%s", ErrArtifactContractMismatch, rule.Role, artifact.SchemaVersion, rule.SchemaVersion)
			}
			var qualified bool
			if err := tx.QueryRow(ctx, `
				SELECT EXISTS (
				  SELECT 1 FROM qualifications
				  WHERE artifact_id=$1 AND qualification_type=$2 AND result='PASS'
				)
			`, artifact.ID, rule.QualificationType).Scan(&qualified); err != nil {
				return err
			}
			if !qualified {
				return fmt.Errorf("%w: role=%s qualification=%s", ErrArtifactNotQualified, rule.Role, rule.QualificationType)
			}
		}
		if len(missing) != 0 {
			return fmt.Errorf("%w: %v", ErrRequiredRolesMissing, missing)
		}
		sort.Strings(requiredRoles)
		sort.Strings(boundRoles)
		snapshot, err := json.Marshal(qualificationSnapshot{
			ProductContractID:     productContractID.String(),
			ProductContractSHA256: contractSHA,
			RequiredRoles:         requiredRoles,
			BoundRoles:            boundRoles,
			AllQualified:          true,
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
			"promotion_id":            promotionID.String(),
			"from_revision_id":        from,
			"to_revision_id":          req.TargetRevisionID.String(),
			"product_contract_id":     productContractID.String(),
			"product_contract_sha256": contractSHA,
			"reason":                  req.Reason,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor, action, subject_id, product_revision_id, payload)
			VALUES ($1,'PRODUCT_REVISION_PROMOTED',$2,$3,$4)
		`, req.RequestedBy, req.SubjectID, req.TargetRevisionID, auditPayload); err != nil {
			return err
		}

		outboxPayload, _ := json.Marshal(map[string]any{
			"promotion_id":            promotionID.String(),
			"product_revision_id":     req.TargetRevisionID.String(),
			"product_contract_sha256": contractSHA,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO outbox_events(aggregate_type, aggregate_id, event_type, payload)
			VALUES ('Subject',$1,'ProductRevisionPromoted',$2)
		`, req.SubjectID, outboxPayload); err != nil {
			return err
		}

		result = PromotionResult{
			PromotionID:    promotionID,
			FromRevision:   from,
			ToRevision:     req.TargetRevisionID,
			NewLockVersion: newLock,
		}
		return nil
	})
	return result, err
}
