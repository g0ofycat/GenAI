import json
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parent / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()