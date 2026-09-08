# MANDATORY RUNNER POLICY — READ BEFORE ANY GITHUB ACTION

> **STOP:** RealSaS CI/scientific Actions must use the user's local self-hosted runner by default. Do **not** use GitHub-hosted runners unless the user explicitly authorizes that exception for the current task.

## Default / required runner

Use:

```yaml
runs-on: [self-hosted, linux, x64, realsas]
```

Known runner: `realsas-wsl-1660ti`.

## Hard rules

- Do **not** use `ubuntu-latest`, `windows-latest`, `macos-latest`, or any other GitHub-hosted runner label for routine RealSaS work.
- If an Action is needed for tests, CI, scientific diagnostics, or compute, route it to the local self-hosted runner above.
- If the local runner is unavailable, fail closed / report the infrastructure issue. Do **not** silently fall back to GitHub-hosted compute.
- GitHub-hosted runners may be used only after explicit user approval for that specific task, with the reason stated in the workflow/commit.
- Direct GitHub API operations that create blobs/trees/commits/refs are not Actions jobs and do not require a runner.
- Avoid branch-push fan-out and unnecessary Actions volume.

## Why this is mandatory

The project has a local runner specifically to avoid unnecessary GitHub Actions usage/quota and to keep scientific execution on the controlled RealSaS environment.

This policy is also reflected in `AGENTS.md` under **Execution environment**. If any workflow contradicts this file, this policy wins unless the user explicitly overrides it for that task.
