package publicproduct

import (
	"context"
	"errors"
	"regexp"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const ProductOwnerHeader = "X-RealSaS-Product-Owner"

var ownerKeySHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var ErrProductOwnerRequired = errors.New("PUBLIC_PRODUCT_OWNER_REQUIRED")

// NormalizeOwnerKeySHA256 validates the non-secret digest derived by the public
// shell from its HttpOnly browser session secret. Platform never needs the raw
// browser secret and never returns this digest to the browser.
func NormalizeOwnerKeySHA256(value string) (string, error) {
	value = strings.TrimSpace(value)
	if !ownerKeySHA256Pattern.MatchString(value) {
		return "", ErrProductOwnerRequired
	}
	return value, nil
}

// RequireOwner deliberately returns ErrCompileNotFound for both unknown ids and
// owner mismatches so the product gateway does not become an ownership oracle.
func RequireOwner(ctx context.Context, pool *pgxpool.Pool, publicID, ownerKeySHA256 string) error {
	if pool == nil {
		return ErrCompileNotFound
	}
	owner, err := NormalizeOwnerKeySHA256(ownerKeySHA256)
	if err != nil {
		return err
	}
	publicID = strings.TrimSpace(publicID)
	if publicID == "" || len(publicID) > 96 {
		return ErrCompileNotFound
	}
	var present bool
	err = pool.QueryRow(ctx, `
		SELECT true
		FROM public_product_compile_sessions
		WHERE public_id=$1 AND owner_key_sha256=$2
	`, publicID, owner).Scan(&present)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrCompileNotFound
	}
	return err
}
