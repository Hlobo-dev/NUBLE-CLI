"""NUBLE - Institutional-Grade AI Investment Research"""

import os
from pathlib import Path
from rich.console import Console

# Load .env file automatically on import
try:
    from dotenv import load_dotenv
    # Try to find .env in current directory or parent directories
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try package directory
        pkg_dir = Path(__file__).parent.parent.parent
        env_path = pkg_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Last resort - just call load_dotenv to search automatically
            load_dotenv()
except ImportError:
    pass  # dotenv not installed

__version__ = "1.0.0"

console = Console()