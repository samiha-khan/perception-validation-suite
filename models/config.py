"""Config loading shared by every stage of the toolkit."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (repo root by default) as a plain dict."""
    cfg_path = Path(path) if path is not None else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path from config.yaml (repo-root-relative) to an absolute Path."""
    p = Path(relative_path)
    return p if p.is_absolute() else REPO_ROOT / p
