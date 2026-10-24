package artifactstore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type S3Config struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	UseTLS    bool
	Bucket    string
	Prefix    string
	Region    string
}

type S3 struct {
	client *minio.Client
	bucket string
	prefix string
}

func NewS3(cfg S3Config) (*S3, error) {
	if strings.TrimSpace(cfg.Endpoint) == "" || strings.TrimSpace(cfg.Bucket) == "" {
		return nil, errors.New("S3 endpoint and bucket are required")
	}
	client, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.UseTLS,
		Region: cfg.Region,
	})
	if err != nil {
		return nil, err
	}
	return &S3{client: client, bucket: cfg.Bucket, prefix: strings.Trim(cfg.Prefix, "/")}, nil
}

func s3NotFound(err error) bool {
	if err == nil {
		return false
	}
	resp := minio.ToErrorResponse(err)
	return resp.StatusCode == 404 || resp.Code == "NoSuchKey" || resp.Code == "NoSuchObject" || resp.Code == "NotFound"
}

type hashCountingReader struct {
	r io.Reader
	h interface{ Write([]byte) (int, error) }
	n int64
}

func (r *hashCountingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	if n > 0 {
		_, _ = r.h.Write(p[:n])
		r.n += int64(n)
	}
	return n, err
}

func (s *S3) PutStream(ctx context.Context, src io.Reader, expectedSHA string, expectedSize int64) (Object, error) {
	if err := ValidateObjectIdentity(expectedSHA, expectedSize); err != nil {
		return Object{}, err
	}
	key, err := CASKey(expectedSHA, s.prefix)
	if err != nil {
		return Object{}, err
	}
	expected := Object{ContentSHA256: expectedSHA, StorageKey: key, SizeBytes: expectedSize}
	info, statErr := s.client.StatObject(ctx, s.bucket, key, minio.StatObjectOptions{})
	if statErr == nil {
		if info.Size != expected.SizeBytes {
			return Object{}, errors.New("S3_CAS_EXISTING_SIZE_MISMATCH")
		}
		if err := copyVerifiedStream(ctx, io.Discard, src, expectedSHA, expectedSize); err != nil {
			return Object{}, err
		}
		if err := s.Verify(ctx, expected); err != nil {
			return Object{}, fmt.Errorf("S3_CAS_EXISTING_CONTENT_HASH_MISMATCH: %w", err)
		}
		return expected, nil
	}
	if !s3NotFound(statErr) {
		return Object{}, statErr
	}

	tempKey := strings.Trim(s.prefix+"/tmp/realsas-upload/"+uuid.NewString(), "/")
	defer func() {
		_ = s.client.RemoveObject(context.Background(), s.bucket, tempKey, minio.RemoveObjectOptions{})
	}()

	h := sha256.New()
	limited := &io.LimitedReader{R: &contextReader{ctx: ctx, r: src}, N: expectedSize}
	counted := &hashCountingReader{r: limited, h: h}
	_, err = s.client.PutObject(
		ctx,
		s.bucket,
		tempKey,
		counted,
		expectedSize,
		minio.PutObjectOptions{
			ContentType: "application/octet-stream",
			UserMetadata: map[string]string{
				"realsas-sha256": expectedSHA,
				"realsas-size":   fmt.Sprintf("%d", expectedSize),
			},
		},
	)
	if err != nil {
		return Object{}, err
	}
	if counted.n != expectedSize || limited.N != 0 {
		return Object{}, fmt.Errorf("artifact size mismatch: got %d want %d", counted.n, expectedSize)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != expectedSHA {
		return Object{}, fmt.Errorf("artifact content hash mismatch: got %s want %s", got, expectedSHA)
	}
	one := make([]byte, 1)
	n, extraErr := (&contextReader{ctx: ctx, r: src}).Read(one)
	if n != 0 {
		return Object{}, fmt.Errorf("artifact size mismatch: got more than %d bytes", expectedSize)
	}
	if extraErr != nil && !errors.Is(extraErr, io.EOF) {
		return Object{}, extraErr
	}
	tempInfo, err := s.client.StatObject(ctx, s.bucket, tempKey, minio.StatObjectOptions{})
	if err != nil {
		return Object{}, err
	}
	if tempInfo.Size != expectedSize {
		return Object{}, fmt.Errorf("S3_TEMP_OBJECT_SIZE_MISMATCH:%d!=%d", tempInfo.Size, expectedSize)
	}

	if info, statErr = s.client.StatObject(ctx, s.bucket, key, minio.StatObjectOptions{}); statErr == nil {
		if info.Size != expected.SizeBytes {
			return Object{}, errors.New("S3_CAS_EXISTING_SIZE_MISMATCH")
		}
		if err := s.Verify(ctx, expected); err != nil {
			return Object{}, fmt.Errorf("S3_CAS_EXISTING_CONTENT_HASH_MISMATCH: %w", err)
		}
		return expected, nil
	} else if !s3NotFound(statErr) {
		return Object{}, statErr
	}

	_, err = s.client.CopyObject(
		ctx,
		minio.CopyDestOptions{Bucket: s.bucket, Object: key},
		minio.CopySrcOptions{Bucket: s.bucket, Object: tempKey},
	)
	if err != nil {
		return Object{}, err
	}
	if err := s.Verify(ctx, expected); err != nil {
		return Object{}, fmt.Errorf("S3_CAS_POST_WRITE_VERIFICATION_FAILED: %w", err)
	}
	return expected, nil
}

func (s *S3) PutBytes(ctx context.Context, data []byte) (Object, error) {
	digest := HashBytes(data)
	return s.PutStream(ctx, bytes.NewReader(data), digest, int64(len(data)))
}

func (s *S3) GetBytes(ctx context.Context, storageKey string) ([]byte, error) {
	obj, err := s.client.GetObject(ctx, s.bucket, storageKey, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}
	defer obj.Close()
	return io.ReadAll(obj)
}

func (s *S3) Verify(ctx context.Context, expected Object) error {
	if err := ValidateObjectIdentity(expected.ContentSHA256, expected.SizeBytes); err != nil {
		return err
	}
	info, err := s.client.StatObject(ctx, s.bucket, expected.StorageKey, minio.StatObjectOptions{})
	if err != nil {
		return err
	}
	if info.Size != expected.SizeBytes {
		return fmt.Errorf("artifact size mismatch: got %d want %d", info.Size, expected.SizeBytes)
	}
	obj, err := s.client.GetObject(ctx, s.bucket, expected.StorageKey, minio.GetObjectOptions{})
	if err != nil {
		return err
	}
	defer obj.Close()
	return verifyReader(ctx, obj, expected.ContentSHA256, expected.SizeBytes)
}
