# GitHub Actions Runner Policy

**MANDATORY:** RealSaS workflows use the local self-hosted runner unless the user explicitly authorizes a GitHub-hosted exception for the current task.

```yaml
runs-on: [self-hosted, linux, x64, realsas]
```

Known runner: `realsas-wsl-1660ti`.

Never substitute `ubuntu-latest`, `windows-latest`, `macos-latest`, or another GitHub-hosted label as a convenience fallback. If the local runner is unavailable, fail closed and report the infrastructure issue.

See `/RUNNER_POLICY.md` and `/AGENTS.md` → **Execution environment**.
