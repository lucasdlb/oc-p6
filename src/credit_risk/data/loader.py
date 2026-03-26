"""Data loader using Polars for efficient handling of large CSV files."""

import polars as pl
from polars import DataFrame, LazyFrame

from credit_risk.config.paths import DATA_FILES
from credit_risk.config.settings import DataConfig


class DataLoader:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def load_lazy(self, name: str) -> LazyFrame:
        path = DATA_FILES[name]
        return pl.scan_csv(path)

    def load(self, name: str) -> DataFrame:
        path = DATA_FILES[name]
        return pl.read_csv(path)

    def load_application_train(self) -> DataFrame:
        return self.load("application_train")

    def load_application_test(self) -> DataFrame:
        return self.load("application_test")

    def load_bureau(self) -> LazyFrame:
        return self.load_lazy("bureau")

    def load_bureau_balance(self) -> LazyFrame:
        return self.load_lazy("bureau_balance")

    def load_previous_application(self) -> LazyFrame:
        return self.load_lazy("previous_application")

    def load_POS_CASH_balance(self) -> LazyFrame:
        return self.load_lazy("POS_CASH_balance")

    def load_credit_card_balance(self) -> LazyFrame:
        return self.load_lazy("credit_card_balance")

    def load_installments_payments(self) -> LazyFrame:
        return self.load_lazy("installments_payments")

    def load_sample_submission(self) -> DataFrame:
        return self.load("sample_submission")

    def load_all_lazy(self) -> dict[str, LazyFrame]:
        return {
            "bureau": self.load_bureau(),
            "bureau_balance": self.load_bureau_balance(),
            "previous_application": self.load_previous_application(),
            "POS_CASH_balance": self.load_POS_CASH_balance(),
            "credit_card_balance": self.load_credit_card_balance(),
            "installments_payments": self.load_installments_payments(),
        }
