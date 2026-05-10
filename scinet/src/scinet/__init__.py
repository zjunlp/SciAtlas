"""SciScholar Python client and CLI package."""

from .client import SciNetClient

SciScholarClient = SciNetClient

__all__ = ["SciNetClient", "SciScholarClient"]
__version__ = "0.1.0"
