from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
def config(path=None):
    with open(path or ROOT / "configs/experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
