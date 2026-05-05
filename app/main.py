from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("server.env")

import os
import platform


def _patch_platform_machine_for_windows() -> None:
    """Avoid Windows WMI hangs in platform helpers used by dependencies."""
    if os.name != "nt":
        return
    arch = os.getenv("PROCESSOR_ARCHITECTURE", "").strip()
    system_name = os.getenv("OS", "Windows_NT")
    system_name = "Windows" if "WINDOWS" in system_name.upper() else "Windows"

    def _safe_machine() -> str:
        return arch or "AMD64"

    def _safe_system() -> str:
        return "Windows"

    platform.machine = _safe_machine  # type: ignore[assignment]
    platform.system = _safe_system  # type: ignore[assignment]


_patch_platform_machine_for_windows()

from backend import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
    