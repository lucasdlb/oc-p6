"""Path configuration for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DATA_RAW = DATA_DIR

DATA_FILES = {
    "application_train": DATA_DIR / "application_train.csv",
    "application_test": DATA_DIR / "application_test.csv",
    "bureau": DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "previous_application": DATA_DIR / "previous_application.csv",
    "POS_CASH_balance": DATA_DIR / "POS_CASH_balance.csv",
    "credit_card_balance": DATA_DIR / "credit_card_balance.csv",
    "installments_payments": DATA_DIR / "installments_payments.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}
