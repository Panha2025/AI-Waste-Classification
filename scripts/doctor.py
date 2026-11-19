"""Quick setup diagnostics; does not reveal bot token or read user photos."""
from pathlib import Path
import importlib.util
import os
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("WasteAI setup doctor")
    print("Python:", sys.version.split()[0])
    print("OS:", platform.platform())
    for module in ("PIL", "numpy", "pandas", "sklearn", "tensorflow", "telegram", "dotenv", "pytest"):
        found = importlib.util.find_spec(module) is not None
        print(f"  {module:12s}: {'OK' if found else 'MISSING'}")
    print("Raw public images:", sum(1 for p in (ROOT / "data/raw/public").rglob("*") if p.is_file() and not p.name.startswith(".")))
    print("Raw local images:", sum(1 for p in (ROOT / "data/raw/local").rglob("*") if p.is_file() and not p.name.startswith(".")))
    print("Split manifest exists:", (ROOT / "data/processed/split_manifest.csv").is_file())
    print("Selected model exists:", (ROOT / "artifacts/model_manifest.json").is_file())
    # Check env file existence, never print its secret values.
    print("Private .env exists:", (ROOT / ".env").is_file())


if __name__ == "__main__":
    main()
