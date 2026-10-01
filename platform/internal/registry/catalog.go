package registry

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type QualifiedCatalog struct {
	Pool              *pgxpool.Pool
	QualificationType string
}

func (c QualifiedCatalog) FindQualified(ctx context.Context, artifactType, schemaVersion, semanticSHA string) (uuid.UUID, bool, error) {
	qualificationType := c.QualificationType
	if qualificationType == "" {
		qualificationType = "REUSE_ELIGIBLE"
	}
	var id uuid.UUID
	err := c.Pool.QueryRow(ctx, `
		SELECT a.id
		FROM artifacts a
		JOIN artifact_types t ON t.id=a.artifact_type_id
		WHERE t.name=$1
		  AND t.schema_version=$2
		  AND a.semantic_sha256=$3
		  AND EXISTS (
		    SELECT 1 FROM qualifications q
		    WHERE q.artifact_id=a.id
		      AND q.qualification_type=$4
		      AND q.result='PASS'
		  )
		LIMIT 1
	`, artifactType, schemaVersion, semanticSHA, qualificationType).Scan(&id)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return uuid.Nil, false, nil
		}
		return uuid.Nil, false, err
	}
	return id, true, nil
}
