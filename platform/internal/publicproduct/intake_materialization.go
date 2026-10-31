package publicproduct

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/registry"
)

const MaxSourceObservationBytes = 64 << 20

func ReadIntakeViewBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	compileID string,
	leaseToken uuid.UUID,
	slot string,
) ([]byte, string, string, error) {
	if store == nil {
		return nil, "", "", errors.New("artifact store is required")
	}
	if err := requireLiveIntakeLease(ctx, pool, compileID, leaseToken); err != nil {
		return nil, "", "", err
	}
	slot = strings.ToLower(strings.TrimSpace(slot))
	if slot != "front" && slot != "back" {
		return nil, "", "", errors.New("PUBLIC_PRODUCT_VIEW_SLOT_INVALID")
	}
	var artifactID uuid.UUID
	var mimeType, originalName string
	if err := pool.QueryRow(ctx, `
		SELECT v.artifact_id,v.mime_type,v.original_name
		FROM public_product_compile_views v
		JOIN public_product_compile_sessions s ON s.id=v.session_id
		WHERE s.public_id=$1 AND v.slot=$2
	`, compileID, slot).Scan(&artifactID, &mimeType, &originalName); err != nil {
		return nil, "", "", err
	}
	item, err := registry.Read(ctx, pool, artifactID)
	if err != nil {
		return nil, "", "", err
	}
	payload, err := store.GetBytes(ctx, item.StorageKey)
	if err != nil {
		return nil, "", "", err
	}
	if int64(len(payload)) != item.SizeBytes {
		return nil, "", "", errors.New("PUBLIC_PRODUCT_VIEW_BYTE_LENGTH_DRIFT")
	}
	return payload, mimeType, originalName, nil
}

func CompleteIntakeBytes(
	ctx context.Context,
	pool *pgxpool.Pool,
	store artifactstore.Store,
	codeSHA string,
	compileID string,
	leaseToken uuid.UUID,
	payload []byte,
	actor string,
) (Status, error) {
	if store == nil {
		return Status{}, errors.New("artifact store is required")
	}
	if len(payload) == 0 || len(payload) > MaxSourceObservationBytes {
		return Status{}, errors.New("PUBLIC_PRODUCT_SOURCE_OBSERVATION_SIZE_INVALID")
	}
	if err := requireLiveIntakeLease(ctx, pool, compileID, leaseToken); err != nil {
		return Status{}, err
	}
	object, err := store.PutBytes(ctx, payload)
	if err != nil {
		return Status{}, err
	}
	registered, err := registry.Import(ctx, pool, store, registry.ImportRequest{
		ArtifactType:  "RealSaS.SourceObservation",
		SchemaVersion: "v1",
		Object:        object,
		SourceURI:     "public-materializer://" + compileID,
		CreatedBy:     actor,
	})
	if err != nil {
		return Status{}, err
	}
	return CompleteIntakeJob(
		ctx,
		pool,
		codeSHA,
		compileID,
		leaseToken,
		registered.ArtifactID,
		actor,
	)
}
