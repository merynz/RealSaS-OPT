package artifactstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestCASKeyMatchesCrossLanguageContract(t *testing.T) {
	digest := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	got, err := CASKey(digest, "realsas")
	if err != nil {
		t.Fatal(err)
	}
	want := "realsas/cas/sha256/01/23/" + digest
	if got != want {
		t.Fatalf("got=%s want=%s", got, want)
	}
}

func TestLocalCASIsImmutableAndVerified(t *testing.T) {
	root := t.TempDir()
	store, err := NewLocal(root)
	if err != nil {
		t.Fatal(err)
	}
	first, err := store.PutBytes(context.Background(), []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.PutBytes(context.Background(), []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	if err := store.Verify(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, filepath.FromSlash(first.StorageKey))
	if err := os.WriteFile(path, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Verify(context.Background(), first); err == nil {
		t.Fatal("expected corruption detection")
	}
}
