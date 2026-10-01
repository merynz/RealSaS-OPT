package outbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type Event struct {
	ID              int64
	ClaimToken      uuid.UUID
	AggregateType   string
	AggregateID     uuid.UUID
	EventType       string
	Payload         map[string]any
	DeliveryAttempt int
}

var ErrClaimDrift = errors.New("outbox claim token drift")

func ClaimBatch(ctx context.Context, pool *pgxpool.Pool, limit int, staleAfter time.Duration) ([]Event, error) {
	if limit < 1 || limit > 500 {
		return nil, errors.New("outbox claim limit must be 1..500")
	}
	if staleAfter <= 0 {
		staleAfter = 5 * time.Minute
	}

	var claimed []Event
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		rows, err := tx.Query(ctx, `
			SELECT id,aggregate_type,aggregate_id,event_type,payload,delivery_attempts
			FROM outbox_events
			WHERE delivered_at IS NULL
			  AND available_at <= now()
			  AND (claimed_at IS NULL OR claimed_at < now() - $1::interval)
			ORDER BY id
			LIMIT $2
			FOR UPDATE SKIP LOCKED
		`, intervalLiteral(staleAfter), limit)
		if err != nil {
			return err
		}
		defer rows.Close()

		var rowsToClaim []Event
		for rows.Next() {
			var item Event
			var payload []byte
			if err := rows.Scan(&item.ID, &item.AggregateType, &item.AggregateID, &item.EventType, &payload, &item.DeliveryAttempt); err != nil {
				return err
			}
			if err := json.Unmarshal(payload, &item.Payload); err != nil {
				return err
			}
			rowsToClaim = append(rowsToClaim, item)
		}
		if err := rows.Err(); err != nil {
			return err
		}

		claimed = claimed[:0]
		for _, item := range rowsToClaim {
			token := uuid.New()
			attempt := item.DeliveryAttempt + 1
			if _, err := tx.Exec(ctx, `
				UPDATE outbox_events
				SET claim_token=$2, claimed_at=now(), delivery_attempts=$3
				WHERE id=$1
			`, item.ID, token, attempt); err != nil {
				return err
			}
			item.ClaimToken = token
			item.DeliveryAttempt = attempt
			claimed = append(claimed, item)
		}
		return nil
	})
	return claimed, err
}

func Ack(ctx context.Context, pool *pgxpool.Pool, event Event) error {
	return persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		result, err := tx.Exec(ctx, `
			UPDATE outbox_events
			SET delivered_at=now(), claim_token=NULL, claimed_at=NULL, last_error=NULL
			WHERE id=$1 AND delivered_at IS NULL AND claim_token=$2
		`, event.ID, event.ClaimToken)
		if err != nil {
			return err
		}
		if result.RowsAffected() != 1 {
			return ErrClaimDrift
		}
		return nil
	})
}

func Fail(ctx context.Context, pool *pgxpool.Pool, event Event, failure error, retryAfter time.Duration) error {
	if retryAfter < 0 {
		return errors.New("retry_after must be non-negative")
	}
	message := "unknown failure"
	if failure != nil {
		message = failure.Error()
	}
	if len(message) > 4000 {
		message = message[:4000]
	}
	return persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		result, err := tx.Exec(ctx, `
			UPDATE outbox_events
			SET last_error=$3,
			    available_at=now() + $4::interval,
			    claim_token=NULL,
			    claimed_at=NULL
			WHERE id=$1 AND delivered_at IS NULL AND claim_token=$2
		`, event.ID, event.ClaimToken, message, intervalLiteral(retryAfter))
		if err != nil {
			return err
		}
		if result.RowsAffected() != 1 {
			return ErrClaimDrift
		}
		return nil
	})
}

func intervalLiteral(d time.Duration) string {
	return fmt.Sprintf("%f seconds", d.Seconds())
}
