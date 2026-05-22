from __future__ import annotations

from pathlib import Path

_src_package = Path(__file__).resolve().parent / "src" / "sciatlas"
if _src_package.exists():
    __path__.insert(0, str(_src_package))

from .client import SciAtlasClient
from .cli import main

__all__ = ["main", "SciAtlasClient"]
