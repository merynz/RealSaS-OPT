package publicproduct

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestNormalizeOwnerKeySHA256(t *testing.T) {
	owner := strings.Repeat("a", 64)
	got, err := NormalizeOwnerKeySHA256(owner)
	if err != nil || got != owner {
		t.Fatalf("owner=%q err=%v", got, err)
	}
	for _, invalid := range []string{"", "abc", strings.Repeat("A", 64), strings.Repeat("0", 63)} {
		if _, err := NormalizeOwnerKeySHA256(invalid); !errors.Is(err, ErrProductOwnerRequired) {
			t.Fatalf("invalid owner %q err=%v", invalid, err)
		}
	}
}

func TestOwnerContextRoundTrip(t *testing.T) {
	owner := strings.Repeat("b", 64)
	ctx, err := WithOwnerKeySHA256(context.Background(), owner)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := OwnerKeySHA256FromContext(ctx)
	if !ok || got != owner {
		t.Fatalf("owner context got=%q ok=%v", got, ok)
	}
	if _, ok := OwnerKeySHA256FromContext(context.Background()); ok {
		t.Fatal("unscoped context unexpectedly has product owner")
	}
}
