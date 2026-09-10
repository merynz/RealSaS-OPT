from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# External research projects may appear in reports/audits, but RealSaS-owned model,
# package, class and architecture identities must not adopt their branding. DINO is
# intentionally absent: it is a direct upstream model dependency and may retain its
# real identity where technically required.
FORBIDDEN_EXTERNAL_BRANDS = (
    "riganything",
    "skintokens",
    "tokenrig",
    "unirig",
    "charactergen",
    "mapanything",
)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def test_model_paths_use_realsas_owned_names() -> None:
    for path in MODELS.rglob("*"):
        if not path.is_file():
            continue
        compact = _compact(str(path.relative_to(MODELS)))
        for brand in FORBIDDEN_EXTERNAL_BRANDS:
            assert brand not in compact, f"third-party brand in RealSaS model path: {path}"


def test_python_model_identifiers_do_not_adopt_external_branding() -> None:
    for path in MODELS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []
        architecture_strings: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("RealSaS."):
                    architecture_strings.append(node.value)
        for value in names + architecture_strings:
            compact = _compact(value)
            for brand in FORBIDDEN_EXTERNAL_BRANDS:
                assert brand not in compact, (
                    f"third-party brand adopted by RealSaS model identifier in {path}: {value}"
                )
