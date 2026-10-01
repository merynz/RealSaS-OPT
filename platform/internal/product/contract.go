package product

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

type RoleRule struct {
	Role              string `json:"role"`
	ArtifactType      string `json:"artifact_type,omitempty"`
	SchemaVersion     string `json:"schema_version,omitempty"`
	QualificationType string `json:"qualification_type"`
	Required          bool   `json:"required"`
}

type Contract struct {
	Name     string         `json:"name"`
	Version  string         `json:"version"`
	Roles    []RoleRule     `json:"roles"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type SealedContract struct {
	ID             uuid.UUID
	ContractSHA256 string
	Reused         bool
}

func (c Contract) normalized() Contract {
	if c.Metadata == nil {
		c.Metadata = map[string]any{}
	}
	c.Roles = append([]RoleRule(nil), c.Roles...)
	sort.Slice(c.Roles, func(i, j int) bool { return c.Roles[i].Role < c.Roles[j].Role })
	return c
}

func (c Contract) Validate() error {
	c = c.normalized()
	if c.Name == "" || c.Version == "" {
		return errors.New("product contract name/version are required")
	}
	if len(c.Roles) == 0 {
		return errors.New("product contract requires at least one role")
	}
	seen := map[string]struct{}{}
	for _, role := range c.Roles {
		if role.Role == "" || role.QualificationType == "" {
			return errors.New("product role and qualification_type are required")
		}
		if _, ok := seen[role.Role]; ok {
			return fmt.Errorf("duplicate product role %s", role.Role)
		}
		seen[role.Role] = struct{}{}
	}
	return nil
}

func (c Contract) SHA256() (string, error) {
	c = c.normalized()
	if err := c.Validate(); err != nil {
		return "", err
	}
	return semantic.JSONSHA256(c)
}

func SealContract(
	ctx context.Context,
	pool *pgxpool.Pool,
	contract Contract,
	createdBy string,
) (SealedContract, error) {
	contract = contract.normalized()
	if pool == nil || createdBy == "" {
		return SealedContract{}, errors.New("pool and created_by are required")
	}
	sha, err := contract.SHA256()
	if err != nil {
		return SealedContract{}, err
	}
	raw, err := json.Marshal(contract)
	if err != nil {
		return SealedContract{}, err
	}
	var out SealedContract
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		return SealedContract{}, err
	}
	defer tx.Rollback(ctx)
	var existing uuid.UUID
	var existingSHA string
	err = tx.QueryRow(ctx, `
		SELECT id,contract_sha256
		FROM product_contracts
		WHERE name=$1 AND version=$2
		FOR UPDATE
	`, contract.Name, contract.Version).Scan(&existing, &existingSHA)
	switch {
	case err == nil:
		if existingSHA != sha {
			return SealedContract{}, errors.New("PRODUCT_CONTRACT_VERSION_IMMUTABILITY_VIOLATION")
		}
		out = SealedContract{ID: existing, ContractSHA256: sha, Reused: true}
	case errors.Is(err, pgx.ErrNoRows):
		id := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO product_contracts
			  (id,name,version,contract_sha256,contract,created_by,sealed_at)
			VALUES ($1,$2,$3,$4,$5,$6,now())
		`, id, contract.Name, contract.Version, sha, raw, createdBy); err != nil {
			return SealedContract{}, err
		}
		out = SealedContract{ID: id, ContractSHA256: sha}
	default:
		return SealedContract{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return SealedContract{}, err
	}
	return out, nil
}

func loadContractTx(ctx context.Context, tx pgx.Tx, id uuid.UUID) (Contract, string, error) {
	var raw []byte
	var sha string
	if err := tx.QueryRow(ctx, `
		SELECT contract,contract_sha256 FROM product_contracts WHERE id=$1
	`, id).Scan(&raw, &sha); err != nil {
		return Contract{}, "", err
	}
	var contract Contract
	if err := json.Unmarshal(raw, &contract); err != nil {
		return Contract{}, "", err
	}
	contract = contract.normalized()
	recomputed, err := contract.SHA256()
	if err != nil {
		return Contract{}, "", err
	}
	if recomputed != sha {
		return Contract{}, "", fmt.Errorf("PRODUCT_CONTRACT_HASH_DRIFT:%s!=%s", recomputed, sha)
	}
	return contract, sha, nil
}

func LoadContract(ctx context.Context, pool *pgxpool.Pool, id uuid.UUID) (Contract, string, error) {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return Contract{}, "", err
	}
	defer tx.Rollback(ctx)
	contract, sha, err := loadContractTx(ctx, tx, id)
	if err != nil {
		return Contract{}, "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return Contract{}, "", err
	}
	return contract, sha, nil
}
