

from pathlib import Path


_API_ROOT = Path(__file__).resolve().parents[4]
def data_dir() -> Path:
    return _API_ROOT / "upload"

