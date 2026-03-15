"""Public package interface for the MAWSR library."""

from .service import WebRepairService
from .dashboard import create_dashboard_app, create_dashboard_blueprint

__all__ = [
    "WebRepairService",
    "create_dashboard_app",
    "create_dashboard_blueprint",
]
