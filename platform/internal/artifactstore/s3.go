package artifactstore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

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

func (s *S3) PutBytes(ctx context.Context, data []byte) (Object, error) {
	digest := HashBytes(data)
	key, err := CASKey(digest, s.prefix)
	if err != nil {
		return Object{}, err
	}
	expected := Object{ContentSHA256: digest, StorageKey: key, SizeBytes: int64(len(data))}
	info, statErr := s.client.StatObject(ctx, s.bucket, key, minio.StatObjectOptions{})
	if statErr == nil {
		if info.Size != expected.SizeBytes {
			return Object{}, errors.New("S3_CAS_EXISTING_SIZE_MISMATCH")
		}
		if err := s.Verify(ctx, expected); err != nil {
			return Object{}, fmt.Errorf("S3_CAS_EXISTING_CONTENT_HASH_MISMATCH: %w", err)
		}
		return expected, nil
	}
	resp := minio.ToErrorResponse(statErr)
	if resp.StatusCode != 404 && resp.Code != "NoSuchKey" && resp.Code != "NoSuchObject" && resp.Code != "NotFound" {
		return Object{}, statErr
	}
	_, err = s.client.PutObject(
		ctx,
		s.bucket,
		key,
		bytes.NewReader(data),
		int64(len(data)),
		minio.PutObjectOptions{
			ContentType:  "application/octet-stream",
			UserMetadata: map[string]string{"realsas-sha256": digest},
		},
	)
	if err != nil {
		return Object{}, err
	}
	if err := s.Verify(ctx, expected); err != nil {
		return Object{}, fmt.Errorf("S3_CAS_POST_WRITE_VERIFICATION_FAILED: %w", err)
	}
	return expected, nil
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
	return verifyReader(obj, expected.ContentSHA256, expected.SizeBytes)
}
