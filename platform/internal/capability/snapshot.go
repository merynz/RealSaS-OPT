package capability

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

type VersionIdentity struct {
	ImplementationSHA256 string `json:"implementation_sha256"`
	PolicySHA256         string `json:"policy_sha256"`
	ParametersSHA256     string `json:"parameters_sha256"`
}

func (v VersionIdentity) Validate() error {
	if err := semantic.ValidateSHA256(v.ImplementationSHA256); err != nil {
		return fmt.Errorf("implementation sha: %w", err)
	}
	if err := semantic.ValidateSHA256(v.PolicySHA256); err != nil {
		return fmt.Errorf("policy sha: %w", err)
	}
	if err := semantic.ValidateSHA256(v.ParametersSHA256); err != nil {
		return fmt.Errorf("parameters sha: %w", err)
	}
	return nil
}

type ReleaseBinding struct {
	Descriptor Descriptor      `json:"descriptor"`
	Version    VersionIdentity `json:"version"`
}

type ReleaseSnapshot struct {
	ReleaseID           uuid.UUID
	CapabilitySetSHA256 string
	Registry            *Registry
	Versions            map[string]VersionIdentity
}

var (
	ErrCapabilitySnapshotAlreadySealed = errors.New("engine release capability snapshot already sealed with different identity")
	ErrCapabilitySnapshotMissing       = errors.New("engine release capability snapshot missing")
)

func SealReleaseSnapshot(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
	bindings []ReleaseBinding,
	createdBy string,
) (string, error) {
	if pool == nil || releaseID == uuid.Nil || createdBy == "" {
		return "", errors.New("pool, release_id and created_by are required")
	}
	if len(bindings) == 0 {
		return "", errors.New("capability snapshot cannot be empty")
	}
	seen := map[string]struct{}{}
	normalized := append([]ReleaseBinding(nil), bindings...)
	for i := range normalized {
		if err := normalized[i].Descriptor.Validate(); err != nil {
			return "", err
		}
		if err := normalized[i].Version.Validate(); err != nil {
			return "", fmt.Errorf("%s: %w", normalized[i].Descriptor.ID, err)
		}
		if _, ok := seen[normalized[i].Descriptor.ID]; ok {
			return "", fmt.Errorf("duplicate capability binding %s", normalized[i].Descriptor.ID)
		}
		seen[normalized[i].Descriptor.ID] = struct{}{}
	}
	sort.Slice(normalized, func(i, j int) bool {
		return normalized[i].Descriptor.ID < normalized[j].Descriptor.ID
	})
	setSHA, err := semantic.JSONSHA256(normalized)
	if err != nil {
		return "", err
	}

	tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	var existing *string
	if err := tx.QueryRow(ctx,
		"SELECT capability_set_sha256 FROM engine_releases WHERE id=$1 FOR UPDATE",
		releaseID,
	).Scan(&existing); err != nil {
		return "", err
	}
	if existing != nil {
		if *existing != setSHA {
			return "", ErrCapabilitySnapshotAlreadySealed
		}
		return setSHA, tx.Commit(ctx)
	}

	for _, binding := range normalized {
		descriptorJSON, err := json.Marshal(binding.Descriptor)
		if err != nil {
			return "", err
		}
		descriptorSHA, err := semantic.JSONSHA256(binding.Descriptor)
		if err != nil {
			return "", err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO engine_release_capabilities
			  (release_id,capability_id,kind,owner_module_id,executor_activity,
			   descriptor_sha256,implementation_sha256,policy_sha256,parameters_sha256,descriptor)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
		`,
			releaseID,
			binding.Descriptor.ID,
			string(binding.Descriptor.Kind),
			binding.Descriptor.OwnerModuleID,
			binding.Descriptor.ExecutorActivity,
			descriptorSHA,
			binding.Version.ImplementationSHA256,
			binding.Version.PolicySHA256,
			binding.Version.ParametersSHA256,
			descriptorJSON,
		); err != nil {
			return "", err
		}
	}
	tag, err := tx.Exec(ctx, `
		UPDATE engine_releases
		SET capability_set_sha256=$2
		WHERE id=$1 AND capability_set_sha256 IS NULL
	`, releaseID, setSHA)
	if err != nil {
		return "", err
	}
	if tag.RowsAffected() != 1 {
		return "", ErrCapabilitySnapshotAlreadySealed
	}
	audit, _ := json.Marshal(map[string]any{
		"engine_release_id":     releaseID.String(),
		"capability_set_sha256": setSHA,
		"capability_count":      len(normalized),
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO audit_events(actor,action,payload)
		VALUES ($1,'ENGINE_RELEASE_CAPABILITY_SET_SEALED',$2)
	`, createdBy, audit); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return setSHA, nil
}

func LoadReleaseSnapshot(
	ctx context.Context,
	pool *pgxpool.Pool,
	releaseID uuid.UUID,
) (ReleaseSnapshot, error) {
	var setSHA *string
	if err := pool.QueryRow(ctx,
		"SELECT capability_set_sha256 FROM engine_releases WHERE id=$1",
		releaseID,
	).Scan(&setSHA); err != nil {
		return ReleaseSnapshot{}, err
	}
	if setSHA == nil || *setSHA == "" {
		return ReleaseSnapshot{}, ErrCapabilitySnapshotMissing
	}
	rows, err := pool.Query(ctx, `
		SELECT capability_id,implementation_sha256,policy_sha256,parameters_sha256,descriptor
		FROM engine_release_capabilities
		WHERE release_id=$1
		ORDER BY capability_id
	`, releaseID)
	if err != nil {
		return ReleaseSnapshot{}, err
	}
	defer rows.Close()
	var descriptors []Descriptor
	versions := map[string]VersionIdentity{}
	for rows.Next() {
		var id, impl, policy, params string
		var raw []byte
		if err := rows.Scan(&id, &impl, &policy, &params, &raw); err != nil {
			return ReleaseSnapshot{}, err
		}
		var descriptor Descriptor
		if err := json.Unmarshal(raw, &descriptor); err != nil {
			return ReleaseSnapshot{}, err
		}
		if descriptor.ID != id {
			return ReleaseSnapshot{}, fmt.Errorf("capability descriptor identity drift: row=%s descriptor=%s", id, descriptor.ID)
		}
		descriptors = append(descriptors, descriptor)
		versions[id] = VersionIdentity{
			ImplementationSHA256: impl,
			PolicySHA256:         policy,
			ParametersSHA256:     params,
		}
	}
	if err := rows.Err(); err != nil {
		return ReleaseSnapshot{}, err
	}
	registry, err := NewRegistry(descriptors)
	if err != nil {
		return ReleaseSnapshot{}, err
	}
	recomputed := make([]ReleaseBinding, 0, len(descriptors))
	for _, descriptor := range descriptors {
		recomputed = append(recomputed, ReleaseBinding{Descriptor: descriptor, Version: versions[descriptor.ID]})
	}
	sort.Slice(recomputed, func(i, j int) bool { return recomputed[i].Descriptor.ID < recomputed[j].Descriptor.ID })
	recomputedSHA, err := semantic.JSONSHA256(recomputed)
	if err != nil {
		return ReleaseSnapshot{}, err
	}
	if recomputedSHA != *setSHA {
		return ReleaseSnapshot{}, fmt.Errorf("CAPABILITY_SET_HASH_DRIFT:%s!=%s", recomputedSHA, *setSHA)
	}
	return ReleaseSnapshot{
		ReleaseID:           releaseID,
		CapabilitySetSHA256: *setSHA,
		Registry:            registry,
		Versions:            versions,
	}, nil
}
