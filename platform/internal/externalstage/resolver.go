package externalstage

import (
	"context"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/merynz/RealSaS-OPT/platform/internal/domain"
)

func ResolverInputs(ctx context.Context, pool *pgxpool.Pool, attemptID uuid.UUID) (map[string][]domain.ArtifactInputIdentity, error) {
	identities, err := LoadIdentities(ctx, pool, attemptID)
	if err != nil {
		return nil, err
	}
	out := make(map[string][]domain.ArtifactInputIdentity, len(identities))
	for stageID, rows := range identities {
		for _, row := range rows {
			out[stageID] = append(out[stageID], domain.ArtifactInputIdentity{
				Role:           row.Role,
				Ordinal:        len(out[stageID]),
				ArtifactType:   row.ArtifactType,
				SemanticSHA256: row.SemanticSHA256,
			})
		}
	}
	return out, nil
}
