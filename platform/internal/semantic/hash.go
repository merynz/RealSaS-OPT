package semantic

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
)

var ErrInvalidSHA256 = errors.New("invalid lowercase sha256")

func CanonicalJSON(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func JSONSHA256(v any) (string, error) {
	b, err := CanonicalJSON(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func ValidateSHA256(v string) error {
	if len(v) != 64 {
		return ErrInvalidSHA256
	}
	raw, err := hex.DecodeString(v)
	if err != nil || len(raw) != sha256.Size {
		return ErrInvalidSHA256
	}
	if hex.EncodeToString(raw) != v {
		return ErrInvalidSHA256
	}
	return nil
}
