package release

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type Purpose string

const (
	PurposeProduct  Purpose = "PRODUCT"
	PurposeResearch Purpose = "RESEARCH"
)

type StageBinding struct {
	Ordinal              int            `json:"ordinal"`
	StageID              string         `json:"stage_id"`
	ImplementationSHA256 string         `json:"implementation_sha256"`
	PolicySHA256         string         `json:"policy_sha256"`
	SemanticParameters   map[string]any `json:"semantic_parameters"`
}

type Manifest struct {
	ContractVersion string         `json:"contract_version"`
	Name            string         `json:"name"`
	Purpose         Purpose        `json:"purpose"`
	Stages          []StageBinding `json:"stages"`
}

type Sealed struct {
	ReleaseID     uuid.UUID
	ReleaseSHA256 string
	Reused        bool
}

type Info struct {
	ID            uuid.UUID
	Name          string
	ReleaseSHA256 string
	Purpose       Purpose
	SealedAt      time.Time
}

var (
	ErrGraphDrift          = errors.New("engine release canonical stage graph drift")
	ErrProductReleaseOnly  = errors.New("product operation requires PRODUCT engine release")
	ErrResearchReleaseOnly = errors.New("research attempt requires RESEARCH engine release")
	ErrReleaseNotFound     = errors.New("engine release not found")
)

func (m Manifest) normalized() Manifest {
	if m.ContractVersion == "" {
		m.ContractVersion = "RealSaS.EngineReleaseManifest.v1"
	}
	for i := range m.Stages {
		if m.Stages[i].SemanticParameters == nil {
			m.Stages[i].SemanticParameters = map[string]any{}
		}
	}
	return m
}

func (m Manifest) Validate(g *stagegraph.Graph) error {
	m = m.normalized()
	if m.Name == "" {
		return errors.New("engine release name is required")
	}
	if m.Purpose != PurposeProduct && m.Purpose != PurposeResearch {
		return errors.New("engine release purpose must be PRODUCT or RESEARCH")
	}
	stages := g.Stages()
	if len(stages) == 0 || len(m.Stages) != len(stages) {
		return ErrGraphDrift
	}
	for i, expected := range stages {
		got := m.Stages[i]
		if got.Ordinal != expected.Ordinal || got.StageID != expected.ID {
			return fmt.Errorf("%w: ordinal=%d got=%s expected=%s", ErrGraphDrift, expected.Ordinal, got.StageID, expected.ID)
		}
		if err := semantic.ValidateSHA256(got.ImplementationSHA256); err != nil {
			return fmt.Errorf("stage %s implementation sha: %w", got.StageID, err)
		}
		if err := semantic.ValidateSHA256(got.PolicySHA256); err != nil {
			return fmt.Errorf("stage %s policy sha: %w", got.StageID, err)
		}
	}
	return nil
}

func (m Manifest) SHA256(g *stagegraph.Graph) (string, error) {
	m = m.normalized()
	if err := m.Validate(g); err != nil {
		return "", err
	}
	return semantic.JSONSHA256(m)
}

func Seal(
	ctx context.Context,
	pool *pgxpool.Pool,
	g *stagegraph.Graph,
	manifest Manifest,
	createdBy string,
) (Sealed, error) {
	manifest = manifest.normalized()
	if createdBy == "" {
		return Sealed{}, errors.New("created_by is required")
	}
	releaseSHA, err := manifest.SHA256(g)
	if err != nil {
		return Sealed{}, err
	}

	var out Sealed
	err = persistence.WithSerializableRetry(ctx, pool, 5, func(tx pgx.Tx) error {
		var existingID uuid.UUID
		var existingName string
		var existingPurpose string
		err := tx.QueryRow(ctx, `
			SELECT id,name,purpose
			FROM engine_releases
			WHERE release_sha256=$1
		`, releaseSHA).Scan(&existingID, &existingName, &existingPurpose)
		switch {
		case err == nil:
			if existingName != manifest.Name || Purpose(existingPurpose) != manifest.Purpose {
				return errors.New("engine release identity metadata drift")
			}
			out = Sealed{ReleaseID: existingID, ReleaseSHA256: releaseSHA, Reused: true}
			return nil
		case !errors.Is(err, pgx.ErrNoRows):
			return err
		}

		id := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
			VALUES ($1,$2,$3,$4,$5,now())
		`, id, manifest.Name, releaseSHA, string(manifest.Purpose), createdBy); err != nil {
			return err
		}

		for _, stage := range manifest.Stages {
			params, err := json.Marshal(stage.SemanticParameters)
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
				INSERT INTO engine_release_stages
				  (release_id,stage_id,ordinal,implementation_sha256,policy_sha256,semantic_parameters)
				VALUES ($1,$2,$3,$4,$5,$6)
			`, id, stage.StageID, stage.Ordinal, stage.ImplementationSHA256, stage.PolicySHA256, params); err != nil {
				return err
			}
		}

		audit, _ := json.Marshal(map[string]any{
			"engine_release_id": id.String(),
			"release_sha256":    releaseSHA,
			"purpose":           string(manifest.Purpose),
			"stage_count":       len(manifest.Stages),
		})
		if _, err := tx.Exec(ctx, `
			INSERT INTO audit_events(actor,action,payload)
			VALUES ($1,'ENGINE_RELEASE_SEALED',$2)
		`, createdBy, audit); err != nil {
			return err
		}
		out = Sealed{ReleaseID: id, ReleaseSHA256: releaseSHA, Reused: false}
		return nil
	})
	return out, err
}

func LoadInfo(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (Info, error) {
	var out Info
	var purpose string
	if err := pool.QueryRow(ctx, `
		SELECT id,name,release_sha256,purpose,sealed_at
		FROM engine_releases
		WHERE id=$1
	`, id).Scan(&out.ID, &out.Name, &out.ReleaseSHA256, &purpose, &out.SealedAt); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Info{}, ErrReleaseNotFound
		}
		return Info{}, err
	}
	out.Purpose = Purpose(purpose)
	return out, nil
}

func LoadVersions(
	ctx context.Context,
	pool *pgxpool.Pool,
	id uuid.UUID,
	requireProduct bool,
) (map[string]StageVersion, error) {
	info, err := LoadInfo(ctx, pool, id)
	if err != nil {
		return nil, err
	}
	if requireProduct && info.Purpose != PurposeProduct {
		return nil, ErrProductReleaseOnly
	}

	rows, err := pool.Query(ctx, `
		SELECT stage_id,implementation_sha256,policy_sha256,semantic_parameters
		FROM engine_release_stages
		WHERE release_id=$1
		ORDER BY ordinal
	`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make(map[string]StageVersion)
	for rows.Next() {
		var stageID, impl, policy string
		var params []byte
		if err := rows.Scan(&stageID, &impl, &policy, &params); err != nil {
			return nil, err
		}
		var decoded any
		if err := json.Unmarshal(params, &decoded); err != nil {
			return nil, err
		}
		paramsSHA, err := semantic.JSONSHA256(decoded)
		if err != nil {
			return nil, err
		}
		out[stageID] = StageVersion{
			ImplementationSHA256: impl,
			PolicySHA256:         policy,
			ParametersSHA256:     paramsSHA,
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, ErrGraphDrift
	}
	return out, nil
}
