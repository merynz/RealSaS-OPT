package control

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
)

func productModelIdentityTx(ctx context.Context, tx pgx.Tx, attemptID uuid.UUID) ([]map[string]any, error) {
	selected, err := modelselection.LoadAttemptTx(ctx, tx, attemptID)
	if err != nil {
		return nil, err
	}
	return modelselection.IdentityRows(selected), nil
}

func bindProductRevisionModelsTx(ctx context.Context, tx pgx.Tx, attemptID, revisionID uuid.UUID) error {
	return modelselection.CopyAttemptToProductRevisionTx(ctx, tx, attemptID, revisionID)
}

func verifyProductRevisionModelsTx(ctx context.Context, tx pgx.Tx, attemptID, revisionID uuid.UUID) error {
	var missingOrDrift bool
	if err := tx.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1
			FROM attempt_model_releases amr
			LEFT JOIN product_revision_model_releases prmr
			  ON prmr.product_revision_id=$2
			 AND prmr.family=amr.family
			 AND prmr.model_release_id=amr.model_release_id
			WHERE amr.attempt_id=$1 AND prmr.product_revision_id IS NULL
		) OR EXISTS (
			SELECT 1
			FROM product_revision_model_releases prmr
			LEFT JOIN attempt_model_releases amr
			  ON amr.attempt_id=$1
			 AND amr.family=prmr.family
			 AND amr.model_release_id=prmr.model_release_id
			WHERE prmr.product_revision_id=$2 AND amr.attempt_id IS NULL
		)
	`, attemptID, revisionID).Scan(&missingOrDrift); err != nil {
		return err
	}
	if missingOrDrift {
		return errors.New("PRODUCT_REVISION_MODEL_BINDING_MISSING_OR_DRIFT")
	}
	return nil
}
