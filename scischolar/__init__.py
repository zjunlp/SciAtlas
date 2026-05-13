from __future__ import annotations

from pathlib import Path

_src_package = Path(__file__).resolve().parent / "src" / "scischolar"
if _src_package.exists():
    __path__.append(str(_src_package))

from .client import SciScholarClient
from .cli import main

__all__ = ["main", "SciScholarClient"]
