package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("command required: stages, attempt, release, research-start, research-run, product-compile, product-render")
	}
	routes := map[string]string{"release": "/v1/releases", "research-start": "/v1/research/attempts", "research-run": "/v1/research/compile", "product-compile": "/v1/product/compile", "product-render": "/v1/product/render"}
	flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
	endpoint := flags.String("api", "http://127.0.0.1:8080", "operator API")
	requestFile := flags.String("request", "-", "JSON request file, or - for stdin")
	id := flags.String("id", "", "attempt UUID")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	method := http.MethodPost
	path, ok := routes[args[0]]
	var body []byte
	if args[0] == "stages" {
		method = http.MethodGet
		path = "/v1/stages"
		ok = true
	}
	if args[0] == "attempt" {
		method = http.MethodGet
		path = "/v1/attempts/" + *id
		ok = *id != ""
	}
	if !ok {
		return fmt.Errorf("unknown command or missing --id: %s", args[0])
	}
	if method == http.MethodPost {
		var err error
		if *requestFile == "-" {
			body, err = io.ReadAll(io.LimitReader(os.Stdin, 8<<20))
		} else {
			body, err = os.ReadFile(*requestFile)
		}
		if err != nil {
			return err
		}
	}
	req, err := http.NewRequest(method, strings.TrimRight(*endpoint, "/")+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 16<<20))
	if err != nil {
		return err
	}
	if response.StatusCode >= 400 {
		return fmt.Errorf("%s: %s", response.Status, strings.TrimSpace(string(data)))
	}
	_, err = os.Stdout.Write(data)
	return err
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
