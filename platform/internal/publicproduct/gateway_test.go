package publicproduct

import (
	"strings"
	"testing"
)

func pngBytes() []byte {
	return []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}
}

func jpegBytes() []byte {
	return []byte{0xff, 0xd8, 0xff, 0xe0}
}

func TestValidateViewsAcceptsFrontAndOptionalBackWithoutInventingViews(t *testing.T) {
	front := ViewUpload{Slot: "front", Name: "front.png", MIMEType: "image/png", BytesBase64: pngBytes()}
	views, err := validateViews([]ViewUpload{front})
	if err != nil {
		t.Fatal(err)
	}
	if len(views) != 1 || views[0].Slot != "front" {
		t.Fatalf("single-view intake drift: %+v", views)
	}

	back := ViewUpload{Slot: "back", Name: "back.jpg", MIMEType: "image/jpeg", BytesBase64: jpegBytes()}
	views, err = validateViews([]ViewUpload{front, back})
	if err != nil {
		t.Fatal(err)
	}
	if len(views) != 2 || views[1].Slot != "back" {
		t.Fatalf("two-view intake drift: %+v", views)
	}
}

func TestValidateViewsRejectsAuthorityFabricationAndSpoofedMedia(t *testing.T) {
	cases := []struct {
		name  string
		views []ViewUpload
		want  string
	}{
		{
			name:  "back without front",
			views: []ViewUpload{{Slot: "back", Name: "back.png", MIMEType: "image/png", BytesBase64: pngBytes()}},
			want:  "FRONT_VIEW_REQUIRED",
		},
		{
			name: "duplicate front",
			views: []ViewUpload{
				{Slot: "front", Name: "a.png", MIMEType: "image/png", BytesBase64: pngBytes()},
				{Slot: "front", Name: "b.png", MIMEType: "image/png", BytesBase64: pngBytes()},
			},
			want: "SLOT_DUPLICATE",
		},
		{
			name:  "spoofed png",
			views: []ViewUpload{{Slot: "front", Name: "front.png", MIMEType: "image/png", BytesBase64: []byte("not a png")}},
			want:  "FORMAT_INVALID",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := validateViews(tc.views)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want substring %s", err, tc.want)
			}
		})
	}
}

func TestPublicStageNeverExposesInternalCapabilityPrefix(t *testing.T) {
	value := "product.mechanics"
	if got := publicStage(&value); got != "MECHANICS" {
		t.Fatalf("stage=%s", got)
	}
	if got := publicStage(nil); got != "INPUT" {
		t.Fatalf("nil stage=%s", got)
	}
}
