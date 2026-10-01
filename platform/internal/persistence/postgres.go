package persistence

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func Open(ctx context.Context, databaseURL string) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse postgres config: %w", err)
	}
	cfg.MaxConns = 20
	cfg.MinConns = 1
	cfg.MaxConnLifetime = 30 * time.Minute
	cfg.MaxConnIdleTime = 5 * time.Minute
	cfg.HealthCheckPeriod = 30 * time.Second
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("open postgres pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	return pool, nil
}

func WithSerializableRetry(
	ctx context.Context,
	pool *pgxpool.Pool,
	maxAttempts int,
	fn func(pgx.Tx) error,
) error {
	if maxAttempts < 1 {
		return errors.New("maxAttempts must be >= 1")
	}
	var last error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
		if err != nil {
			return fmt.Errorf("begin serializable tx: %w", err)
		}
		err = fn(tx)
		if err != nil {
			_ = tx.Rollback(ctx)
			if isRetryableTxError(err) && attempt < maxAttempts {
				last = err
				continue
			}
			return err
		}
		if err = tx.Commit(ctx); err != nil {
			if isRetryableTxError(err) && attempt < maxAttempts {
				last = err
				continue
			}
			return fmt.Errorf("commit serializable tx: %w", err)
		}
		return nil
	}
	return fmt.Errorf("serializable transaction retries exhausted: %w", last)
}

func isRetryableTxError(err error) bool {
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return false
	}
	return pgErr.Code == "40001" || pgErr.Code == "40P01"
}
