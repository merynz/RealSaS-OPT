package artifactstore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const MaxArtifactObjectBytes int64 = 8 << 30

type Object struct {
	ContentSHA256 string `json:"content_sha256"`
	StorageKey    string `json:"storage_key"`
	SizeBytes     int64  `json:"size_bytes"`
}

type Store interface {
	PutStream(context.Context, io.Reader, string, int64) (Object, error)
	PutBytes(context.Context, []byte) (Object, error)
	GetBytes(context.Context, string) ([]byte, error)
	Verify(context.Context, Object) error
}

func ValidateSHA256(value string) error {
	if len(value) != 64 {
		return errors.New("invalid sha256 length")
	}
	raw, err := hex.DecodeString(value)
	if err != nil || len(raw) != sha256.Size || hex.EncodeToString(raw) != value {
		return errors.New("invalid lowercase sha256")
	}
	return nil
}

func ValidateObjectIdentity(contentSHA256 string, sizeBytes int64) error {
	if err := ValidateSHA256(contentSHA256); err != nil {
		return err
	}
	if sizeBytes < 0 {
		return errors.New("artifact size must be non-negative")
	}
	if sizeBytes > MaxArtifactObjectBytes {
		return fmt.Errorf("artifact size exceeds %d bytes", MaxArtifactObjectBytes)
	}
	return nil
}

func CASKey(contentSHA256, prefix string) (string, error) {
	if err := ValidateSHA256(contentSHA256); err != nil {
		return "", err
	}
	base := fmt.Sprintf("cas/sha256/%s/%s/%s", contentSHA256[:2], contentSHA256[2:4], contentSHA256)
	prefix = strings.Trim(prefix, "/")
	if prefix == "" {
		return base, nil
	}
	return prefix + "/" + base, nil
}

func HashBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r *contextReader) Read(p []byte) (int, error) {
	select {
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	default:
		return r.r.Read(p)
	}
}

func copyVerifiedStream(ctx context.Context, dst io.Writer, src io.Reader, expectedSHA string, expectedSize int64) error {
	if err := ValidateObjectIdentity(expectedSHA, expectedSize); err != nil {
		return err
	}
	h := sha256.New()
	limited := &io.LimitedReader{R: &contextReader{ctx: ctx, r: src}, N: expectedSize + 1}
	n, err := io.Copy(io.MultiWriter(dst, h), limited)
	if err != nil {
		return err
	}
	if n != expectedSize {
		return fmt.Errorf("artifact size mismatch: got %d want %d", n, expectedSize)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != expectedSHA {
		return fmt.Errorf("artifact content hash mismatch: got %s want %s", got, expectedSHA)
	}
	return nil
}

func verifyReader(ctx context.Context, r io.Reader, expectedSHA string, expectedSize int64) error {
	return copyVerifiedStream(ctx, io.Discard, r, expectedSHA, expectedSize)
}

type Local struct {
	Root string
}

func NewLocal(root string) (*Local, error) {
	if strings.TrimSpace(root) == "" {
		return nil, errors.New("local artifact root is required")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(absolute, 0o755); err != nil {
		return nil, err
	}
	return &Local{Root: absolute}, nil
}

func (s *Local) pathForKey(key string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(key))
	if clean == "." || filepath.IsAbs(clean) || strings.HasPrefix(clean, ".."+string(filepath.Separator)) || clean == ".." {
		return "", errors.New("invalid artifact storage key")
	}
	target := filepath.Join(s.Root, clean)
	rel, err := filepath.Rel(s.Root, target)
	if err != nil || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." {
		return "", errors.New("artifact storage key escapes root")
	}
	return target, nil
}

func (s *Local) PutStream(ctx context.Context, src io.Reader, expectedSHA string, expectedSize int64) (Object, error) {
	if err := ValidateObjectIdentity(expectedSHA, expectedSize); err != nil {
		return Object{}, err
	}
	key, err := CASKey(expectedSHA, "")
	if err != nil {
		return Object{}, err
	}
	path, err := s.pathForKey(key)
	if err != nil {
		return Object{}, err
	}
	expected := Object{ContentSHA256: expectedSHA, StorageKey: key, SizeBytes: expectedSize}
	if _, err := os.Stat(path); err == nil {
		if err := copyVerifiedStream(ctx, io.Discard, src, expectedSHA, expectedSize); err != nil {
			return Object{}, err
		}
		if err := s.Verify(ctx, expected); err != nil {
			return Object{}, fmt.Errorf("CAS_HASH_COLLISION_OR_CORRUPTION: %w", err)
		}
		return expected, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return Object{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return Object{}, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".realsas-cas-stream-*")
	if err != nil {
		return Object{}, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := copyVerifiedStream(ctx, tmp, src, expectedSHA, expectedSize); err != nil {
		_ = tmp.Close()
		return Object{}, err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return Object{}, err
	}
	if err := tmp.Close(); err != nil {
		return Object{}, err
	}
	if _, err := os.Stat(path); err == nil {
		if err := s.Verify(ctx, expected); err != nil {
			return Object{}, fmt.Errorf("CAS_HASH_COLLISION_OR_CORRUPTION: %w", err)
		}
		return expected, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return Object{}, err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return Object{}, err
	}
	if err := s.Verify(ctx, expected); err != nil {
		return Object{}, fmt.Errorf("CAS_POST_WRITE_VERIFICATION_FAILED: %w", err)
	}
	return expected, nil
}

func (s *Local) PutBytes(ctx context.Context, data []byte) (Object, error) {
	digest := HashBytes(data)
	return s.PutStream(ctx, bytes.NewReader(data), digest, int64(len(data)))
}

func (s *Local) GetBytes(_ context.Context, storageKey string) ([]byte, error) {
	path, err := s.pathForKey(storageKey)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(path)
}

func (s *Local) Verify(ctx context.Context, obj Object) error {
	if err := ValidateObjectIdentity(obj.ContentSHA256, obj.SizeBytes); err != nil {
		return err
	}
	path, err := s.pathForKey(obj.StorageKey)
	if err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return verifyReader(ctx, f, obj.ContentSHA256, obj.SizeBytes)
}
