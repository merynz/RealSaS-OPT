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
)

type RequalificationResultSession struct {
	SessionID uuid.UUID
	CompileID string
}

// CreateRequalificationResultSession projects a newly sealed ProductRevision
// back into the existing public workspace lane. It copies only the immutable
// user ownership and SourceObservation support identity from the base compile;
// the new session points at the new requalification Attempt/ProductRevision and
// receives its own opaque compile id and workspace materialization job.
func CreateRequalificationResultSession(
	ctx context.Context,
	pool *pgxpool.Pool,
	baseSessionID uuid.UUID,
	requalificationAttemptID uuid.UUID,
	resultProductRevisionID uuid.UUID,
	actor string,
) (RequalificationResultSession, error) {
	actor = strings.TrimSpace(actor)
	if pool == nil || baseSessionID == uuid.Nil || requalificationAttemptID == uuid.Nil ||
		resultProductRevisionID == uuid.Nil || actor == "" {
		return RequalificationResultSession{}, errors.New("pool, base session, requalification attempt, result revision and actor are required")
	}

	var out RequalificationResultSession
	err := persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var existingID uuid.UUID
		var existingPublicID string
		err := tx.QueryRow(ctx, `
			SELECT id,public_id FROM public_product_compile_sessions
			WHERE attempt_id=$1
		`, requalificationAttemptID).Scan(&existingID, &existingPublicID)
		if err == nil {
			var existingRevision uuid.UUID
			if err := tx.QueryRow(ctx, `
				SELECT product_revision_id FROM public_product_compile_sessions WHERE id=$1
			`, existingID).Scan(&existingRevision); err != nil {
				return err
			}
			if existingRevision != resultProductRevisionID {
				return errors.New("PUBLIC_PRODUCT_REQUALIFICATION_SESSION_REVISION_DRIFT")
			}
			out = RequalificationResultSession{SessionID: existingID, CompileID: existingPublicID}
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		var subjectID, releaseID, sourceID, baseRevisionID uuid.UUID
		var channelLock int64
		var owner string
		var baseState string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,engine_release_id,release_channel_lock_version,
			       source_observation_artifact_id,product_revision_id,owner_key_sha256,state
			FROM public_product_compile_sessions
			WHERE id=$1 FOR SHARE
		`, baseSessionID).Scan(
			&subjectID, &releaseID, &channelLock, &sourceID, &baseRevisionID, &owner, &baseState,
		); err != nil {
			return err
		}
		if baseState != "PASS" || strings.TrimSpace(owner) == "" {
			return errors.New("PUBLIC_PRODUCT_REQUALIFICATION_QUALIFIED_OWNED_BASE_REQUIRED")
		}

		var attemptSubject, attemptRelease uuid.UUID
		var attemptState, attemptKind string
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,engine_release_id,final_state,kind
			FROM attempts WHERE id=$1 FOR SHARE
		`, requalificationAttemptID).Scan(
			&attemptSubject, &attemptRelease, &attemptState, &attemptKind,
		); err != nil {
			return err
		}
		if attemptSubject != subjectID || attemptRelease != releaseID ||
			attemptState != "QUALIFIED" || attemptKind != "requalification_candidate" {
			return errors.New("PUBLIC_PRODUCT_REQUALIFICATION_ATTEMPT_DRIFT")
		}
		var revisionSubject, revisionAttempt uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT subject_id,created_from_attempt_id FROM product_revisions WHERE id=$1 FOR SHARE
		`, resultProductRevisionID).Scan(&revisionSubject, &revisionAttempt); err != nil {
			return err
		}
		if revisionSubject != subjectID || revisionAttempt != requalificationAttemptID || resultProductRevisionID == baseRevisionID {
			return errors.New("PUBLIC_PRODUCT_REQUALIFICATION_PRODUCT_REVISION_DRIFT")
		}

		sessionID := uuid.New()
		publicID := "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
		if _, err := tx.Exec(ctx, `
			INSERT INTO public_product_compile_sessions(
				id,public_id,subject_id,engine_release_id,release_channel_lock_version,
				source_observation_artifact_id,attempt_id,product_revision_id,
				state,current_capability,owner_key_sha256
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'RUNNING','workspace.materialization',$9)
		`, sessionID, publicID, subjectID, releaseID, channelLock, sourceID,
			requalificationAttemptID, resultProductRevisionID, owner); err != nil {
			return err
		}
		payload, _ := json.Marshal(map[string]any{
			"base_compile_session_id":    baseSessionID,
			"base_product_revision_id":   baseRevisionID,
			"result_product_revision_id": resultProductRevisionID,
			"public_id":                  publicID,
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,subject_id,attempt_id,payload)
			VALUES ($1,'PUBLIC_PRODUCT_REQUALIFICATION_SESSION_CREATED',$2,$3,$4)
		`, actor, subjectID, requalificationAttemptID, payload); err != nil {
			return err
		}
		out = RequalificationResultSession{SessionID: sessionID, CompileID: publicID}
		return nil
	})
	if err != nil {
		return RequalificationResultSession{}, err
	}
	if err := ensureWorkspaceMaterialization(ctx, pool, out.SessionID, resultProductRevisionID); err != nil {
		return RequalificationResultSession{}, err
	}
	return out, nil
}
