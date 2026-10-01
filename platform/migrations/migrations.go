package migrations

import (
	"context"
	"database/sql"
	"embed"
	"io/fs"

	"github.com/pressly/goose/v3"
)

//go:embed sql/*.sql
var migrationFS embed.FS

func NewProvider(db *sql.DB) (*goose.Provider, error) {
	sub, err := fs.Sub(migrationFS, "sql")
	if err != nil {
		return nil, err
	}
	return goose.NewProvider(
		goose.DialectPostgres,
		db,
		sub,
		goose.WithTableName("realsas_schema_migrations"),
	)
}

func Up(ctx context.Context, db *sql.DB) error {
	provider, err := NewProvider(db)
	if err != nil {
		return err
	}
	defer provider.Close()
	_, err = provider.Up(ctx)
	return err
}

func Reset(ctx context.Context, db *sql.DB) error {
	provider, err := NewProvider(db)
	if err != nil {
		return err
	}
	defer provider.Close()
	_, err = provider.DownTo(ctx, 0)
	return err
}

func Status(ctx context.Context, db *sql.DB) ([]*goose.MigrationStatus, error) {
	provider, err := NewProvider(db)
	if err != nil {
		return nil, err
	}
	defer provider.Close()
	return provider.Status(ctx)
}
