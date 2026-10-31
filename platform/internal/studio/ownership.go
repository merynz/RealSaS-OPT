package studio

import (
	"context"
	"errors"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/publicproduct"
)

// RequireProjectOwner verifies the public browser/session ownership inherited
// from the source compile. Owner mismatch deliberately collapses to
// STUDIO_PROJECT_NOT_FOUND so the service gateway cannot be used as a project
// existence oracle.
func RequireProjectOwner(ctx context.Context, pool *pgxpool.Pool, projectID, ownerKeySHA256 string) error {
	projectID = strings.TrimSpace(projectID)
	if pool == nil || !projectIDPattern.MatchString(projectID) {
		return ErrProjectNotFound
	}
	owner, err := publicproduct.NormalizeOwnerKeySHA256(ownerKeySHA256)
	if err != nil {
		return err
	}
	var present bool
	err = pool.QueryRow(ctx, `
		SELECT true
		FROM studio_projects p
		JOIN public_product_compile_sessions s ON s.id=p.source_compile_session_id
		WHERE p.public_id=$1 AND s.owner_key_sha256=$2
	`, projectID, owner).Scan(&present)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrProjectNotFound
	}
	return err
}
