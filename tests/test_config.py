import os
import tempfile
from pathlib import Path
from skippy.config import load_config, AppConfig

def test_load_config_missing():
    # Should not raise, should return defaults
    cfg = load_config("/non/existent/path/default.toml")
    assert isinstance(cfg, AppConfig)
    assert cfg.localai.base_url == "http://localhost:8080"

def test_load_config_invalid_toml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("invalid = [toml")
        f_path = f.name

    try:
        cfg = load_config(f_path)
        assert isinstance(cfg, AppConfig)
        assert cfg.localai.base_url == "http://localhost:8080"
    finally:
        os.remove(f_path)

def test_load_config_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("[localai]\nbase_url = 'http://test:1234'")
        f_path = f.name

    try:
        cfg = load_config(f_path)
        assert cfg.localai.base_url == "http://test:1234"
    finally:
        os.remove(f_path)
