package publicproduct

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

var ErrNoIntakeJob = errors.New("PUBLIC_PRODUCT_INTAKE_JOB_NOT_AVAILABLE")

type IntakeView struct {
	Slot       string    `json:"slot"`
	ArtifactID uuid.UUID `json:"artifact_id"`
	MIMEType   string    `json:"mime_type"`
	Name       string    `json:"name"`
}

type IntakeLease struct {
	CompileID   string       `json:"compile_id"`
	LeaseToken  uuid.UUID    `json:"lease_token"`
	LeaseOwner  string       `json:"lease_owner"`
	LeaseExpiry time.Time    `json:"lease_expires_at"`
	Views       []IntakeView `json:"views"`
	Attempt     int          `json:"attempt"`
}

func ClaimIntakeJob(
	ctx context.Context,
	pool *pgxpool.Pool,
	owner string,
	leaseDuration time.Duration,
) (IntakeLease, error) {
	owner = strings.TrimSpace(owner)
	if pool == nil || owner == "" {
		return IntakeLease{}, errors.New("pool and lease owner are required")
	}
	if leaseDuration < time.Minute || leaseDuration > 2*time.Hour {
		return IntakeLease{}, errors.New("lease duration must be between one minute and two hours")
	}

	var lease IntakeLease
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var sessionID uuid.UUID
		var publicID string
		var attemptCount int
		err := tx.QueryRow(ctx, `
			SELECT j.session_id,s.public_id,j.attempt_count
			FROM public_product_intake_jobs j
			JOIN public_product_compile_sessions s ON s.id=j.session_id
			WHERE s.state='WAITING_SOURCE_MATERIALIZATION'
			  AND (
			    j.state='QUEUED'
			    OR (j.state='LEASED' AND j.lease_expires_at <= now())
			  )
			ORDER BY j.created_at,j.session_id
			FOR UPDATE OF j SKIP LOCKED
			LIMIT 1
		`).Scan(&sessionID, &publicID, &attemptCount)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNoIntakeJob
		}
		if err != nil {
			return err
		}

		leaseToken := uuid.New()
		leaseExpiry := time.Now().UTC().Add(leaseDuration)
		if _, err := tx.Exec(ctx, `
			UPDATE public_product_intake_jobs
			SET state='LEASED',lease_token=$2,lease_owner=$3,lease_expires_at=$4,
			    attempt_count=attempt_count+1,last_error_code=NULL
			WHERE session_id=$1
		`, sessionID, leaseToken, owner, leaseExpiry); err != nil {
			return err
		}

		rows, err := tx.Query(ctx, `
			SELECT slot,artifact_id,mime_type,original_name
			FROM public_product_compile_views
			WHERE session_id=$1
			ORDER BY CASE slot WHEN 'front' THEN 0 ELSE 1 END,slot
		`, sessionID)
		if err != nil {
			return err
		}
		defer rows.Close()
		views := make([]IntakeView, 0, 2)
		for rows.Next() {
			var view IntakeView
			if err := rows.Scan(&view.Slot, &view.ArtifactID, &view.MIMEType, &view.Name); err != nil {
				return err
			}
			views = append(views, view)
		}
		if err := rows.Err(); err != nil {
			return err
		}
		if len(views) < 1 || len(views) > 2 {
			return errors.New("PUBLIC_PRODUCT_INTAKE_VIEW_BINDING_DRIFT")
		}
		lease = IntakeLease{
			CompileID:   publicID,
			LeaseToken:  leaseToken,
			LeaseOwner:  owner,
			LeaseExpiry: leaseExpiry,
			Views:       views,
			Attempt:     attemptCount + 1,
		}
		return nil
	})
	return lease, err
}

func CompleteIntakeJob(
	ctx context.Context,
	pool *pgxpool.Pool,
	codeSHA string,
	compileID string,
	leaseToken uuid.UUID,
	sourceObservationID uuid.UUID,
	actor string,
) (Status, error) {
	if leaseToken == uuid.Nil {
		return Status{}, errors.New("lease token is required")
	}
	if err := requireLiveIntakeLease(ctx, pool, compileID, leaseToken); err != nil {
		return Status{}, err
	}
	status, err := AttachSourceObservation(ctx, pool, codeSHA, compileID, sourceObservationID, actor)
	if err != nil {
		return Status{}, err
	}
	if _, err := pool.Exec(ctx, `
		UPDATE public_product_intake_jobs j
		SET state='PASS',lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error_code=NULL
		FROM public_product_compile_sessions s
		WHERE j.session_id=s.id AND s.public_id=$1 AND j.lease_token=$2
	`, compileID, leaseToken); err != nil {
		return Status{}, err
	}
	return status, nil
}

func FailIntakeJob(
	ctx context.Context,
	pool *pgxpool.Pool,
	compileID string,
	leaseToken uuid.UUID,
	errorCode string,
	retry bool,
) error {
	errorCode = strings.TrimSpace(errorCode)
	if pool == nil || strings.TrimSpace(compileID) == "" || leaseToken == uuid.Nil || errorCode == "" {
		return errors.New("pool, compile_id, lease_token and error_code are required")
	}
	if len(errorCode) > 200 {
		return errors.New("error_code exceeds 200 characters")
	}
	return persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var sessionID uuid.UUID
		var currentToken uuid.UUID
		var expiresAt time.Time
		if err := tx.QueryRow(ctx, `
			SELECT j.session_id,j.lease_token,j.lease_expires_at
			FROM public_product_intake_jobs j
			JOIN public_product_compile_sessions s ON s.id=j.session_id
			WHERE s.public_id=$1 AND j.state='LEASED'
			FOR UPDATE OF j
		`, compileID).Scan(&sessionID, &currentToken, &expiresAt); err != nil {
			return err
		}
		if currentToken != leaseToken || time.Now().UTC().After(expiresAt) {
			return errors.New("PUBLIC_PRODUCT_INTAKE_LEASE_INVALID_OR_EXPIRED")
		}
		if retry {
			_, err := tx.Exec(ctx, `
				UPDATE public_product_intake_jobs
				SET state='QUEUED',lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error_code=$2
				WHERE session_id=$1
			`, sessionID, errorCode)
			return err
		}
		if _, err := tx.Exec(ctx, `
			UPDATE public_product_intake_jobs
			SET state='FAILED',lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error_code=$2
			WHERE session_id=$1
		`, sessionID, errorCode); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
			UPDATE public_product_compile_sessions
			SET state='FAILED',error_code=$2,current_capability='product.input'
			WHERE id=$1
		`, sessionID, errorCode)
		return err
	})
}

func requireLiveIntakeLease(
	ctx context.Context,
	pool *pgxpool.Pool,
	compileID string,
	leaseToken uuid.UUID,
) error {
	if pool == nil || strings.TrimSpace(compileID) == "" || leaseToken == uuid.Nil {
		return errors.New("pool, compile_id and lease_token are required")
	}
	var currentToken uuid.UUID
	var expiresAt time.Time
	if err := pool.QueryRow(ctx, `
		SELECT j.lease_token,j.lease_expires_at
		FROM public_product_intake_jobs j
		JOIN public_product_compile_sessions s ON s.id=j.session_id
		WHERE s.public_id=$1 AND j.state='LEASED'
	`, compileID).Scan(&currentToken, &expiresAt); err != nil {
		return err
	}
	if currentToken != leaseToken || time.Now().UTC().After(expiresAt) {
		return errors.New("PUBLIC_PRODUCT_INTAKE_LEASE_INVALID_OR_EXPIRED")
	}
	return nil
}
