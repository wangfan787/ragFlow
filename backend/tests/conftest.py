"""测试隔离：公共 YAML 默认值 + 临时数据目录，不读取本机密钥。"""

import pytest
from backend.src.config.settings import Settings, settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    config = Settings(local_path=None, overrides={"data_dir": str(tmp_path / "data")})
    monkeypatch.setattr(settings, "_data", config._data)
