package publicproduct

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func BindProductionRelease(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	codeSHA string,
	actor string,
) (ReleaseChannel, error) {
	if pool == nil || releaseID == uuid.Nil || strings.TrimSpace(codeSHA) == "" || strings.TrimSpace(actor) == "" {
		return ReleaseChannel{}, errors.New("pool, release_id, code_sha and actor are required")
	}
	if err := release.RequireCanonicalMain(ctx, pool, releaseID, codeSHA); err != nil {
		return ReleaseChannel{}, err
	}
	var purpose string
	if err := pool.QueryRow(ctx, `SELECT purpose FROM engine_releases WHERE id=$1`, releaseID).Scan(&purpose); err != nil {
		return ReleaseChannel{}, err
	}
	if purpose != string(release.PurposeProduct) {
		return ReleaseChannel{}, errors.New("PUBLIC_PRODUCT_RELEASE_MUST_BE_PRODUCT")
	}

	var out ReleaseChannel
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
			INSERT INTO product_release_channels(channel,engine_release_id,lock_version,updated_by,updated_at)
			VALUES ('PRODUCTION',$1,0,$2,now())
			ON CONFLICT (channel) DO UPDATE SET
				engine_release_id=EXCLUDED.engine_release_id,
				lock_version=product_release_channels.lock_version+1,
				updated_by=EXCLUDED.updated_by,
				updated_at=now()
		`, releaseID, actor); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, `
			SELECT channel,engine_release_id,lock_version
			FROM product_release_channels WHERE channel='PRODUCTION'
		`).Scan(&out.Channel, &out.EngineReleaseID, &out.LockVersion); err != nil {
			return err
		}
		payload, _ := json.Marshal(map[string]any{
			"channel": ProductionChannel, "engine_release_id": releaseID, "lock_version": out.LockVersion,
		})
		_, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,payload)
			VALUES ($1,'PUBLIC_PRODUCT_RELEASE_CHANNEL_BOUND',$2)
		`, actor, payload)
		return err
	})
	return out, err
}

func ActiveProductionRelease(ctx context.Context, pool *pgxpool.Pool) (ReleaseChannel, error) {
	if pool == nil {
		return ReleaseChannel{}, errors.New("pool is required")
	}
	var out ReleaseChannel
	if err := pool.QueryRow(ctx, `
		SELECT channel,engine_release_id,lock_version
		FROM product_release_channels
		WHERE channel='PRODUCTION'
	`).Scan(&out.Channel, &out.EngineReleaseID, &out.LockVersion); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return ReleaseChannel{}, ErrReleaseChannelUnbound
		}
		return ReleaseChannel{}, err
	}
	return out, nil
}
