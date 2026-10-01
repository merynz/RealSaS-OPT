package dispatch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/outbox"
)

type Publisher interface {
	Publish(ctx context.Context, commandType string, payload map[string]any) error
}

type Result struct {
	Claimed   int
	Delivered int
	Failed    int
}

func DispatchOnce(ctx context.Context, pool *pgxpool.Pool, publisher Publisher, limit int) (Result, error) {
	if publisher == nil {
		return Result{}, errors.New("publisher is required")
	}
	events, err := outbox.ClaimBatch(ctx, pool, limit, 5*time.Minute)
	if err != nil {
		return Result{}, err
	}
	result := Result{Claimed: len(events)}
	for _, event := range events {
		commandType, payload, err := loadCommand(ctx, pool, event)
		if err == nil {
			err = publisher.Publish(ctx, commandType, payload)
		}
		if err != nil {
			if failErr := outbox.Fail(ctx, pool, event, err, retryDelay(event.DeliveryAttempt)); failErr != nil {
				return result, fmt.Errorf("record outbox failure after %v: %w", err, failErr)
			}
			result.Failed++
			continue
		}
		if err := outbox.Ack(ctx, pool, event); err != nil {
			return result, err
		}
		result.Delivered++
	}
	return result, nil
}

func loadCommand(ctx context.Context, pool *pgxpool.Pool, event outbox.Event) (string, map[string]any, error) {
	if event.AggregateType != "Command" || event.AggregateID == uuid.Nil {
		return "", nil, errors.New("outbox aggregate is not a command")
	}
	var commandType string
	var raw []byte
	if err := pool.QueryRow(ctx, `
		SELECT command_type,payload
		FROM commands
		WHERE id=$1
	`, event.AggregateID).Scan(&commandType, &raw); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return "", nil, errors.New("outbox command not found")
		}
		return "", nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return "", nil, err
	}
	return commandType, payload, nil
}

func retryDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	if attempt > 8 {
		attempt = 8
	}
	return time.Duration(1<<uint(attempt)) * time.Second
}
