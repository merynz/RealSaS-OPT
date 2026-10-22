package release

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

type SourceAuthority string

const (
	SourceAuthorityCanonicalMain SourceAuthority = "CANONICAL_MAIN"
	SourceAuthorityLegacyFrozen  SourceAuthority = "LEGACY_FROZEN"
)

type SourceBinding struct {
	ReleaseID uuid.UUID       `json:"release_id"`
	CodeSHA   string          `json:"code_sha"`
	Authority SourceAuthority `json:"authority"`
	BoundBy   string          `json:"bound_by"`
	BoundAt   time.Time       `json:"bound_at"`
	Reused    bool            `json:"reused"`
}

var (
	ErrCanonicalSourceRequired = errors.New("ENGINE_RELEASE_CANONICAL_MAIN_SOURCE_REQUIRED")
	ErrSourceAuthorityDrift    = errors.New("ENGINE_RELEASE_SOURCE_AUTHORITY_DRIFT")
)

func validateGitSHA(value string) error {
	if len(value) != 40 || strings.ToLower(value) != value {
		return errors.New("git source sha must be 40 lowercase hex characters")
	}
	_, err := hex.DecodeString(value)
	return err
}

func bindSource(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	codeSHA string,
	authority SourceAuthority,
	actor string,
) (SourceBinding, error) {
	codeSHA = strings.TrimSpace(codeSHA)
	actor = strings.TrimSpace(actor)
	if releaseID == uuid.Nil || actor == "" {
		return SourceBinding{}, errors.New("release_id and actor are required")
	}
	if err := validateGitSHA(codeSHA); err != nil {
		return SourceBinding{}, err
	}
	if authority != SourceAuthorityCanonicalMain && authority != SourceAuthorityLegacyFrozen {
		return SourceBinding{}, errors.New("source authority must be CANONICAL_MAIN or LEGACY_FROZEN")
	}

	out := SourceBinding{ReleaseID: releaseID, CodeSHA: codeSHA, Authority: authority, BoundBy: actor}
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var release uuid.UUID
		if err := tx.QueryRow(ctx, "SELECT id FROM engine_releases WHERE id=$1 FOR UPDATE", releaseID).Scan(&release); err != nil {
			return fmt.Errorf("engine release: %w", err)
		}

		if authority == SourceAuthorityCanonicalMain {
			var canonicalCodeSHA string
			err := tx.QueryRow(ctx, `
				SELECT code_sha
				FROM engine_release_source_bindings
				WHERE release_id=$1 AND authority='CANONICAL_MAIN'
			`, releaseID).Scan(&canonicalCodeSHA)
			switch {
			case err == nil && canonicalCodeSHA != codeSHA:
				return ErrSourceAuthorityDrift
			case err != nil && !errors.Is(err, pgx.ErrNoRows):
				return err
			}
		}

		var existingAuthority string
		var boundBy string
		var boundAt time.Time
		err := tx.QueryRow(ctx, `
			SELECT authority,bound_by,bound_at
			FROM engine_release_source_bindings
			WHERE release_id=$1 AND code_sha=$2
		`, releaseID, codeSHA).Scan(&existingAuthority, &boundBy, &boundAt)
		switch {
		case err == nil:
			if SourceAuthority(existingAuthority) != authority {
				return ErrSourceAuthorityDrift
			}
			out.Authority = SourceAuthority(existingAuthority)
			out.BoundBy = boundBy
			out.BoundAt = boundAt
			out.Reused = true
			return nil
		case !errors.Is(err, pgx.ErrNoRows):
			return err
		}

		if _, err := tx.Exec(ctx, `
			INSERT INTO engine_release_source_bindings
			  (release_id,code_sha,authority,bound_by)
			VALUES ($1,$2,$3,$4)
		`, releaseID, codeSHA, string(authority), actor); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, `
			SELECT bound_at FROM engine_release_source_bindings
			WHERE release_id=$1 AND code_sha=$2
		`, releaseID, codeSHA).Scan(&out.BoundAt); err != nil {
			return err
		}
		audit, _ := json.Marshal(map[string]any{
			"engine_release_id": releaseID.String(),
			"code_sha":          codeSHA,
			"authority":         authority,
		})
		_, err = tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,payload)
			VALUES ($1,'ENGINE_RELEASE_SOURCE_BOUND',$2)
		`, actor, audit)
		return err
	})
	return out, err
}

// BindCanonicalMain records that this exact EngineRelease is executable from
// exactly one main SHA. EngineRelease manifests do not include Git code SHA, so
// the first canonical binding completes executable source identity. Rebinding the
// same release to a different main SHA is source-authority drift and fails closed.
//
// The API is the only normal caller: clients do not get to self-assert
// CANONICAL_MAIN provenance. Source authority is deliberately independent from
// product authority. A release becoming executable from canonical main does not
// select a ModelRelease channel, qualify an artifact, promote a ProductRevision,
// or otherwise change shipping behavior. Those transitions remain explicit
// operations in their own domains.
func BindCanonicalMain(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	deployedCodeSHA string,
	actor string,
) (SourceBinding, error) {
	return bindSource(ctx, pool, releaseID, deployedCodeSHA, SourceAuthorityCanonicalMain, actor)
}

// BindLegacyFrozen exists only for explicit migration/provenance tooling. It
// intentionally cannot satisfy RequireCanonicalMain, so a legacy branch can be
// preserved and inspected without becoming legal source for a new execution.
func BindLegacyFrozen(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	legacyCodeSHA string,
	actor string,
) (SourceBinding, error) {
	return bindSource(ctx, pool, releaseID, legacyCodeSHA, SourceAuthorityLegacyFrozen, actor)
}

func RequireCanonicalMain(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	deployedCodeSHA string,
) error {
	if releaseID == uuid.Nil {
		return ErrCanonicalSourceRequired
	}
	if err := validateGitSHA(strings.TrimSpace(deployedCodeSHA)); err != nil {
		return ErrCanonicalSourceRequired
	}
	var exists bool
	if err := pool.QueryRow(ctx, `
		SELECT EXISTS(
			SELECT 1 FROM engine_release_source_bindings
			WHERE release_id=$1 AND code_sha=$2 AND authority='CANONICAL_MAIN'
		)
	`, releaseID, deployedCodeSHA).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return ErrCanonicalSourceRequired
	}
	return nil
}

func SourceBindings(ctx context.Context, pool *pgxpool.Pool, releaseID uuid.UUID) ([]SourceBinding, error) {
	rows, err := pool.Query(ctx, `
		SELECT release_id,code_sha,authority,bound_by,bound_at
		FROM engine_release_source_bindings
		WHERE release_id=$1
		ORDER BY bound_at,code_sha
	`, releaseID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]SourceBinding, 0)
	for rows.Next() {
		var row SourceBinding
		var authority string
		if err := rows.Scan(&row.ReleaseID, &row.CodeSHA, &authority, &row.BoundBy, &row.BoundAt); err != nil {
			return nil, err
		}
		row.Authority = SourceAuthority(authority)
		out = append(out, row)
	}
	return out, rows.Err()
}
