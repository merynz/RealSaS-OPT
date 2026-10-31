package publicproduct

import (
	"bytes"
	"errors"
	"strings"
)

func validateViews(input []ViewUpload) ([]ViewUpload, error) {
	if len(input) < 1 || len(input) > 2 {
		return nil, errors.New("PUBLIC_PRODUCT_REQUIRES_ONE_OR_TWO_SOURCE_VIEWS")
	}
	seen := map[string]struct{}{}
	front := false
	out := make([]ViewUpload, 0, len(input))
	for _, view := range input {
		view.Slot = strings.ToLower(strings.TrimSpace(view.Slot))
		view.MIMEType = strings.ToLower(strings.TrimSpace(view.MIMEType))
		view.Name = strings.TrimSpace(view.Name)
		if view.Slot != "front" && view.Slot != "back" {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SLOT_INVALID")
		}
		if _, duplicate := seen[view.Slot]; duplicate {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SLOT_DUPLICATE")
		}
		seen[view.Slot] = struct{}{}
		if view.Slot == "front" {
			front = true
		}
		if view.Name == "" || len(view.Name) > 300 {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_NAME_INVALID")
		}
		if len(view.BytesBase64) == 0 || len(view.BytesBase64) > maxPublicViewBytes {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_SIZE_INVALID")
		}
		if !validImageBytes(view.MIMEType, view.BytesBase64) {
			return nil, errors.New("PUBLIC_PRODUCT_VIEW_FORMAT_INVALID")
		}
		out = append(out, view)
	}
	if !front {
		return nil, errors.New("PUBLIC_PRODUCT_FRONT_VIEW_REQUIRED")
	}
	return out, nil
}

func validImageBytes(mimeType string, data []byte) bool {
	switch mimeType {
	case "image/png":
		return len(data) >= 8 && bytes.Equal(data[:8], []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	case "image/jpeg":
		return len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff
	case "image/webp":
		return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP"
	default:
		return false
	}
}
