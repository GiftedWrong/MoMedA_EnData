"""Мелкие общие помощники для всех скриптов."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Читает config.yaml, пути разрешаются относительно корня проекта."""
    try:
        import yaml  # type: ignore
    except ImportError:
        # запасной парсер «key: value», если yaml не стоит
        cfg = {}
        for line in (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
        return cfg
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    return {k: str(v) for k, v in cfg.items()}


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def project_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
