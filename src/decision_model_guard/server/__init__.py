"""HTTP API for Decision Model Guard (`pip install 'decision-model-guard[server]'`)."""
from .app import create_app

__all__ = ["create_app"]
