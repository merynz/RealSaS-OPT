package publicproduct

import (
	"errors"

	"github.com/google/uuid"
)

const (
	ProductionChannel  = "PRODUCTION"
	maxPublicViewBytes = 6 << 20
)

var (
	ErrReleaseChannelUnbound = errors.New("PUBLIC_PRODUCT_RELEASE_CHANNEL_UNBOUND")
	ErrCompileNotFound       = errors.New("PUBLIC_PRODUCT_COMPILE_NOT_FOUND")
)

type ViewUpload struct {
	Slot        string `json:"slot"`
	Name        string `json:"name"`
	MIMEType    string `json:"mime_type"`
	BytesBase64 []byte `json:"bytes_base64"`
}

type StartRequest struct {
	DisplayName string       `json:"display_name,omitempty"`
	Views       []ViewUpload `json:"views"`
}

type StartResponse struct {
	CompileID string `json:"compile_id"`
	Status    string `json:"status"`
	Stage     string `json:"stage"`
	Detail    string `json:"detail"`
	Terminal  bool   `json:"terminal"`
}

type Status struct {
	CompileID   string `json:"compile_id"`
	Status      string `json:"status"`
	Stage       string `json:"stage"`
	Detail      string `json:"detail"`
	Terminal    bool   `json:"terminal"`
	ResultReady bool   `json:"result_ready"`
}

type ReleaseChannel struct {
	Channel         string    `json:"channel"`
	EngineReleaseID uuid.UUID `json:"engine_release_id"`
	LockVersion     int64     `json:"lock_version"`
}

type sessionRecord struct {
	ID                  uuid.UUID
	PublicID            string
	SubjectID           uuid.UUID
	EngineReleaseID     uuid.UUID
	ChannelLockVersion  int64
	SourceObservationID *uuid.UUID
	AttemptID           *uuid.UUID
	ProductRevisionID   *uuid.UUID
	State               string
	CurrentCapability   *string
	ErrorCode           *string
}
