"""本地文档与原图路径；读取路径不创建目录，不依赖启动目录。"""

from pathlib import Path

from .settings import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    configured = Path(settings.text("MVP_DATA_DIR") or "data/md-rag").expanduser()
    return (configured if configured.is_absolute() else REPO_ROOT / configured).resolve()


def uploads_dir() -> Path:
    return data_dir() / "uploads"


def database_path() -> Path:
    return data_dir() / "documents.sqlite3"
