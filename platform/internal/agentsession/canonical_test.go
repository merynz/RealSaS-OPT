package agentsession

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCanonicalCheckoutRejectsReviewOverlayAndStaleMain(t *testing.T) {
	root, origin := t.TempDir(), t.TempDir()
	git := func(dir string, args ...string) string {
		t.Helper()
		c := exec.Command("git", append([]string{"-C", dir}, args...)...)
		data, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v %s", args, err, data)
		}
		return strings.TrimSpace(string(data))
	}
	git(origin, "init", "--bare")
	git(root, "init", "-b", "main")
	git(root, "config", "user.email", "fixture@example.invalid")
	git(root, "config", "user.name", "Fixture")
	file := filepath.Join(root, "source.txt")
	os.WriteFile(file, []byte("one"), 0600)
	git(root, "add", ".")
	git(root, "commit", "-m", "one")
	git(root, "remote", "add", "origin", origin)
	git(root, "push", "origin", "main")
	first := git(root, "rev-parse", "HEAD")
	check := func(want string) {
		t.Helper()
		got, err := CanonicalCheckout(context.Background(), root)
		if want == "" {
			if err != nil || got != first {
				t.Fatalf("canonical %q %v", got, err)
			}
		} else if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("wanted %s: %v", want, err)
		}
	}
	check("")
	git(root, "switch", "-c", "review/source")
	check("REVIEW_BRANCH")
	git(root, "switch", "main")
	os.WriteFile(file, []byte("overlay"), 0600)
	check("SOURCE_OVERLAY")
	git(root, "restore", file)
	os.Mkdir(filepath.Join(root, "compiler"), 0700)
	extra := filepath.Join(root, "compiler", "injected.py")
	os.WriteFile(extra, []byte("pass"), 0600)
	check("UNTRACKED_SOURCE_OVERLAY")
	os.Remove(extra)
	git(root, "switch", "--detach", first)
	check("")
	git(root, "switch", "main")
	os.WriteFile(file, []byte("two"), 0600)
	git(root, "add", ".")
	git(root, "commit", "-m", "two")
	git(root, "push", "origin", "main")
	git(root, "switch", "--detach", first)
	check("NOT_CURRENT_MAIN")
}
