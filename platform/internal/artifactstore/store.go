package artifactstore

import (
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

type Object struct {
	ContentSHA256 string `json:"content_sha256"`
	StorageKey    string `json:"storage_key"`
	SizeBytes     int64  `json:"size_bytes"`
}

type Store interface {
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

func verifyReader(r io.Reader, expectedSHA string, expectedSize int64) error {
	if err := ValidateSHA256(expectedSHA); err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(h, r)
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

func (s *Local) PutBytes(_ context.Context, data []byte) (Object, error) {
	digest := HashBytes(data)
	key, err := CASKey(digest, "")
	if err != nil {
		return Object{}, err
	}
	path, err := s.pathForKey(key)
	if err != nil {
		return Object{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return Object{}, err
	}
	if existing, err := os.ReadFile(path); err == nil {
		if HashBytes(existing) != digest || len(existing) != len(data) {
			return Object{}, errors.New("CAS_HASH_COLLISION_OR_CORRUPTION")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return Object{}, err
	} else {
		tmp, err := os.CreateTemp(filepath.Dir(path), ".realsas-cas-*")
		if err != nil {
			return Object{}, err
		}
		tmpName := tmp.Name()
		defer os.Remove(tmpName)
		if _, err := tmp.Write(data); err != nil {
			tmp.Close()
			return Object{}, err
		}
		if err := tmp.Sync(); err != nil {
			tmp.Close()
			return Object{}, err
		}
		if err := tmp.Close(); err != nil {
			return Object{}, err
		}
		if err := os.Rename(tmpName, path); err != nil {
			return Object{}, err
		}
	}
	obj := Object{ContentSHA256: digest, StorageKey: key, SizeBytes: int64(len(data))}
	if err := s.Verify(context.Background(), obj); err != nil {
		return Object{}, fmt.Errorf("CAS_POST_WRITE_VERIFICATION_FAILED: %w", err)
	}
	return obj, nil
}

func (s *Local) GetBytes(_ context.Context, storageKey string) ([]byte, error) {
	path, err := s.pathForKey(storageKey)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(path)
}

func (s *Local) Verify(_ context.Context, obj Object) error {
	path, err := s.pathForKey(obj.StorageKey)
	if err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return verifyReader(f, obj.ContentSHA256, obj.SizeBytes)
}
