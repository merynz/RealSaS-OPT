package outbox

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestClaimAckAndRetryLifecycle(t *testing.T) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	if _, err := pool.Exec(ctx, "UPDATE outbox_events SET delivered_at=now() WHERE delivered_at IS NULL"); err != nil {
		t.Fatal(err)
	}
	firstAggregate := uuid.New()
	secondAggregate := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload)
		VALUES ('Command',$1,'CompileSubjectRequested','{}'),
		       ('Command',$2,'RenderProductRequested','{}')
	`, firstAggregate, secondAggregate); err != nil {
		t.Fatal(err)
	}

	claims, err := ClaimBatch(ctx, pool, 10, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if len(claims) != 2 || claims[0].DeliveryAttempt != 1 || claims[1].DeliveryAttempt != 1 {
		t.Fatalf("claims=%+v", claims)
	}
	if err := Ack(ctx, pool, claims[0]); err != nil {
		t.Fatal(err)
	}
	if err := Fail(ctx, pool, claims[1], errors.New("temporary"), 0); err != nil {
		t.Fatal(err)
	}

	retry, err := ClaimBatch(ctx, pool, 10, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if len(retry) != 1 || retry[0].ID != claims[1].ID || retry[0].DeliveryAttempt != 2 {
		t.Fatalf("retry=%+v", retry)
	}
	if err := Ack(ctx, pool, retry[0]); err != nil {
		t.Fatal(err)
	}
	remaining, err := ClaimBatch(ctx, pool, 10, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if len(remaining) != 0 {
		t.Fatalf("remaining=%+v", remaining)
	}
}
