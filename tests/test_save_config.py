import os
import tempfile
from pathlib import Path
from skippy.config import load_config, AppConfig

def test_save_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("[localai]\nbase_url = 'http://original:8080'")
        f_path = Path(f.name)

    try:
        cfg = load_config(f_path)
        cfg.localai.base_url = "http://updated:1234"
        cfg.audio.volume = 0.5
        cfg.save()

        # Reload and check
        cfg2 = load_config(f_path)
        assert cfg2.localai.base_url == "http://updated:1234"
        assert cfg2.audio.volume == 0.5
    finally:
        if f_path.exists():
            os.remove(f_path)
