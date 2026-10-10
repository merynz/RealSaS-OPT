package agentsession

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// CanonicalCheckout checks source review never becomes a second execution line.
// Remote main is resolved now, rather than trusting a stale local origin/main.
func CanonicalCheckout(ctx context.Context, root string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	git := func(args ...string) (string, error) {
		command := exec.CommandContext(ctx, "git", append([]string{"-C", root}, args...)...)
		bytes, err := command.Output()
		if err != nil {
			return "", fmt.Errorf("AGENT_CANONICAL_CHECK_FAILED: %w", err)
		}
		return strings.TrimSpace(string(bytes)), nil
	}
	head, err := git("rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	remote, err := git("ls-remote", "--exit-code", "origin", "refs/heads/main")
	if err != nil {
		return "", err
	}
	fields := strings.Fields(remote)
	if len(fields) != 2 || ValidateCode(head, fields[0]) != nil {
		return "", fmt.Errorf("AGENT_CHECKOUT_NOT_CURRENT_MAIN")
	}
	branch, err := git("rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return "", err
	}
	if branch != "main" && branch != "HEAD" {
		return "", fmt.Errorf("AGENT_REVIEW_BRANCH_CANNOT_EXECUTE")
	}
	dirty, err := git("status", "--porcelain", "--untracked-files=no")
	if err != nil {
		return "", err
	}
	if dirty != "" {
		return "", fmt.Errorf("AGENT_SOURCE_OVERLAY_FORBIDDEN")
	}
	untracked, err := git("ls-files", "--others", "--exclude-standard", "--", "compiler", "tools", "platform")
	if err != nil {
		return "", err
	}
	for _, path := range strings.Split(untracked, "\n") {
		switch filepath.Ext(path) {
		case ".py", ".go", ".so", ".dll", ".sh", ".json", ".yaml", ".yml":
			return "", fmt.Errorf("AGENT_UNTRACKED_SOURCE_OVERLAY_FORBIDDEN: %s", path)
		}
	}
	return head, nil
}
