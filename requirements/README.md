# Environment profiles

These files define **current repository development/CI profiles**, not retroactive scientific-environment authority.

- `mainline-ci.txt` — pinned CPU dependencies used by current fast mainline regressions.
- `torch-cpu.txt` — pinned CPU PyTorch profile installed from the official PyTorch CPU wheel index.
- `dev.txt` — pinned repository hygiene/development tools in addition to `mainline-ci.txt`.

Scientific experiments remain governed by their own preregistration, preflight, environment snapshot, checkpoint and result evidence. A root dependency update must never silently rewrite the environment identity of a sealed experiment.

Recommended local setup:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/mainline-ci.txt
python -m pip install -r requirements/torch-cpu.txt
python -m pip install -r requirements/dev.txt
pre-commit install
```

Windows PowerShell activation is `.venv\Scripts\Activate.ps1`.
