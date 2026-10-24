package artifactstore

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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

func TestLocalPutStreamPublishesOnlyVerifiedIdentity(t *testing.T) {
	root := t.TempDir()
	store, err := NewLocal(root)
	if err != nil {
		t.Fatal(err)
	}
	data := "streamed model bytes"
	sha := HashBytes([]byte(data))
	obj, err := store.PutStream(context.Background(), strings.NewReader(data), sha, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if obj.ContentSHA256 != sha || obj.SizeBytes != int64(len(data)) {
		t.Fatalf("stream object drift: %+v", obj)
	}
	if err := store.Verify(context.Background(), obj); err != nil {
		t.Fatal(err)
	}

	wrongSHA := HashBytes([]byte("different"))
	if _, err := store.PutStream(context.Background(), strings.NewReader(data), wrongSHA, int64(len(data))); err == nil {
		t.Fatal("expected hash mismatch")
	}
	wrongKey, _ := CASKey(wrongSHA, "")
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(wrongKey))); !os.IsNotExist(err) {
		t.Fatalf("hash-mismatched object was published: %v", err)
	}

	shortSHA := HashBytes([]byte(data[:5]))
	if _, err := store.PutStream(context.Background(), strings.NewReader(data), shortSHA, 5); err == nil {
		t.Fatal("expected oversized stream rejection")
	}
	shortKey, _ := CASKey(shortSHA, "")
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(shortKey))); !os.IsNotExist(err) {
		t.Fatalf("size-mismatched object was published: %v", err)
	}
}

func TestLocalPutStreamIdempotencyStillValidatesIncomingBytes(t *testing.T) {
	store, err := NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	data := "same immutable object"
	sha := HashBytes([]byte(data))
	first, err := store.PutStream(context.Background(), strings.NewReader(data), sha, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.PutStream(context.Background(), strings.NewReader(data), sha, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("idempotent stream object drift: first=%+v second=%+v", first, second)
	}
	if _, err := store.PutStream(context.Background(), strings.NewReader("wrong request bytes"), sha, int64(len(data))); err == nil {
		t.Fatal("existing CAS object bypassed incoming-body verification")
	}
}

func TestLocalPutStreamCancellationCannotPublish(t *testing.T) {
	root := t.TempDir()
	store, err := NewLocal(root)
	if err != nil {
		t.Fatal(err)
	}
	data := "cancelled upload"
	sha := HashBytes([]byte(data))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.PutStream(ctx, strings.NewReader(data), sha, int64(len(data))); err == nil {
		t.Fatal("expected cancelled upload to fail")
	}
	key, _ := CASKey(sha, "")
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(key))); !os.IsNotExist(err) {
		t.Fatalf("cancelled object was published: %v", err)
	}
}

func TestArtifactObjectLimitAdmitsRecoveredMIRARuntime(t *testing.T) {
	const miraRuntimeBytes int64 = 558840979
	if MaxArtifactObjectBytes < miraRuntimeBytes {
		t.Fatalf("artifact cap %d blocks recovered MIRA runtime %d", MaxArtifactObjectBytes, miraRuntimeBytes)
	}
	if err := ValidateObjectIdentity(strings.Repeat("a", 64), MaxArtifactObjectBytes+1); err == nil {
		t.Fatal("expected bounded artifact cap to reject oversized object")
	}
}
