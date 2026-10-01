package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"github.com/merynz/RealSaS-OPT/platform/migrations"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatal("usage: realsas-migrate <up|reset|status>")
	}
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		log.Fatal("REALSAS_DATABASE_URL is required")
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		log.Fatal(err)
	}
	switch os.Args[1] {
	case "up":
		err = migrations.Up(ctx, db)
	case "reset":
		err = migrations.Reset(ctx, db)
	case "status":
		var rows any
		rows, err = migrations.Status(ctx, db)
		if err == nil {
			fmt.Printf("%v\n", rows)
		}
	default:
		log.Fatalf("unknown command %q", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}
