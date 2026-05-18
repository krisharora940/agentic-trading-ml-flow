from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
EXPERIMENTS_DIR = ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
