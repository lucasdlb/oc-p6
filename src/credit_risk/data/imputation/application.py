"""Application table imputer."""

from __future__ import annotations

import polars as pl
from lightgbm import LGBMRegressor
from typing import override

from polars import DataFrame
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from credit_risk.data.base import ProcessingStep

APARTMENT_COLS = [
    "APARTMENTS_AVG",
    "APARTMENTS_MEDI",
    "APARTMENTS_MODE",
    "BASEMENTAREA_AVG",
    "BASEMENTAREA_MEDI",
    "BASEMENTAREA_MODE",
    "COMMONAREA_AVG",
    "COMMONAREA_MEDI",
    "COMMONAREA_MODE",
    "ELEVATORS_AVG",
    "ELEVATORS_MEDI",
    "ELEVATORS_MODE",
    "ENTRANCES_AVG",
    "ENTRANCES_MEDI",
    "ENTRANCES_MODE",
    "FLOORSMAX_AVG",
    "FLOORSMAX_MEDI",
    "FLOORSMAX_MODE",
    "FLOORSMIN_AVG",
    "FLOORSMIN_MEDI",
    "FLOORSMIN_MODE",
    "LANDAREA_AVG",
    "LANDAREA_MEDI",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_AVG",
    "LIVINGAREA_MEDI",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_AVG",
    "NONLIVINGAREA_MEDI",
    "NONLIVINGAREA_MODE",
    "TOTALAREA_MODE",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_AVG",
    "YEARS_BUILD_MEDI",
    "YEARS_BUILD_MODE",
]

AMT_REQ_CREDIT_BUREAU_COLS = [
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


class ApplicationImputer(ProcessingStep):
    """Imputer for application_train/test tables.

    Domain-aware imputation:
    - OWN_CAR_AGE: -1 when FLAG_OWN_CAR = "N", median when "Y"
    - AMT_REQ_CREDIT_BUREAU_*: 0 when all co-missing
    - Apartment features: 0 when HOUSETYPE_MODE is NaN, median by housing type otherwise
    - EXT_SOURCE_*: median + binary missingness flag
    - OCCUPATION_TYPE: mode by NAME_INCOME_TYPE

    Implements fit/transform to prevent data leakage.
    """

    def __init__(self) -> None:
        self._median_car_age: float = 0.0
        self._apartment_medians: dict[str, float] = {}
        self._apartment_by_housing: dict[str, dict[str, float]] = {}
        self._ext_source_medians: dict[str, float] = {}
        self._occupation_by_income: dict[str, str] = {}
        self._default_occupation: str = "Laborers"
        self._cat_modes: dict[str, str] = {}
        self._int_cols: dict[str, object] = {}
        self._apartment_learned_cols: list[str] = []
        self._amt_req_cols: list[str] = []
        self._high_missing: list[str] = []
        self._low_missing: list[str] = []
        self._iterative_imputer: IterativeImputer | None = None

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_fitted") and self._fitted

    @override
    def fit(self, X: DataFrame, y=None) -> "ApplicationImputer":
        if "OWN_CAR_AGE" in X.columns and "FLAG_OWN_CAR" in X.columns:
            median = (
                X.filter(pl.col("FLAG_OWN_CAR") == "Y")
                .select(pl.col("OWN_CAR_AGE").median())
                .item()
            )
            self._median_car_age = median if median is not None else 0.0

        self._apartment_medians: dict[str, float] = {}
        self._apartment_by_housing: dict[str, dict[str, float]] = {}
        for col in [c for c in APARTMENT_COLS if c in X.columns]:
            self._apartment_medians[col] = X.select(pl.col(col).median()).item() or 0.0
            by_housing: dict[str, float] = {}
            for ht in ["block of flats", "specific housing", "terraced house"]:
                val = X.filter(pl.col("HOUSETYPE_MODE") == ht).select(pl.col(col).median()).item()
                if val is not None:
                    by_housing[ht] = val
            self._apartment_by_housing[col] = by_housing

        self._ext_source_medians: dict[str, float] = {}
        for col in [c for c in EXT_SOURCE_COLS if c in X.columns]:
            self._ext_source_medians[col] = X.select(pl.col(col).median()).item() or 0.0

        self._occupation_by_income: dict[str, str] = {}
        for income in X["NAME_INCOME_TYPE"].unique().to_list():
            if income is None:
                continue
            filtered = X.filter(
                (pl.col("NAME_INCOME_TYPE") == income) & (pl.col("OCCUPATION_TYPE").is_not_null())
            )
            if filtered.height > 0:
                mode_series = filtered.select(pl.col("OCCUPATION_TYPE").drop_nulls().mode().first())
                self._occupation_by_income[income] = mode_series.item()

        if "OCCUPATION_TYPE" in X.columns:
            global_mode = X.select(pl.col("OCCUPATION_TYPE").drop_nulls().mode().first()).item()
            self._default_occupation = global_mode if global_mode is not None else "Laborers"
        else:
            self._default_occupation = "Laborers"

        self._cat_modes: dict[str, str] = {}
        for col in X.columns:
            if X.schema[col] == pl.String:
                mode = X.select(pl.col(col).drop_nulls().mode().first()).item()
                if mode is not None:
                    self._cat_modes[col] = mode

        self._int_cols: dict[str, object] = {
            col: dtype
            for col, dtype in zip(X.columns, X.dtypes, strict=True)
            if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        }

        self._apartment_learned_cols = [c for c in APARTMENT_COLS if c in X.columns]
        self._amt_req_cols = [c for c in AMT_REQ_CREDIT_BUREAU_COLS if c in X.columns]

        pdf = X.drop(self._apartment_learned_cols).to_pandas()
        num_cols = pdf.select_dtypes(include="number").columns.tolist()
        self._high_missing = [c for c in num_cols if X[c].null_count() / len(X) > 0.05]
        self._low_missing = [c for c in num_cols if c not in self._high_missing]

        if self._high_missing:
            self._iterative_imputer = IterativeImputer(
                estimator=LGBMRegressor(n_estimators=50, verbosity=-1),
                max_iter=3,
                random_state=42,
            )
            self._iterative_imputer.fit(pdf[self._high_missing])
        else:
            self._iterative_imputer = None

        self._fitted = True

        return self

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        if "OWN_CAR_AGE" in X.columns and "FLAG_OWN_CAR" in X.columns:
            X = X.with_columns(
                pl.when(pl.col("FLAG_OWN_CAR") == "N")
                .then(pl.lit(-1.0))
                .when(pl.col("OWN_CAR_AGE").is_null())
                .then(pl.lit(self._median_car_age))
                .otherwise(pl.col("OWN_CAR_AGE"))
                .alias("OWN_CAR_AGE")
            )

        if self._amt_req_cols:
            all_null_expr = pl.col(self._amt_req_cols[0]).is_null()
            for col in self._amt_req_cols[1:]:
                all_null_expr = all_null_expr & pl.col(col).is_null()
            fill_exprs = [pl.col(c).fill_null(-1).alias(c) for c in self._amt_req_cols]
            X = X.with_columns(*fill_exprs, all_null_expr.alias("AMT_REQ_CREDIT_BUREAU_missing"))

        for col in self._apartment_learned_cols:
            global_median = self._apartment_medians.get(col, 0.0)
            by_housing = self._apartment_by_housing.get(col, {})
            if by_housing:
                expr = pl.when(pl.col("HOUSETYPE_MODE").is_null()).then(pl.lit(0.0))
                for ht, med in by_housing.items():
                    expr = expr.when(pl.col("HOUSETYPE_MODE") == ht).then(pl.lit(med))
                expr = expr.otherwise(pl.col(col).fill_null(global_median))
                X = X.with_columns(expr.alias(col))
            else:
                X = X.with_columns(pl.col(col).fill_null(global_median).alias(col))

        for col, median in self._ext_source_medians.items():
            if col in X.columns:
                X = X.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(f"{col}_missing"),
                    pl.col(col).fill_null(median).alias(col),
                )

        if "OCCUPATION_TYPE" in X.columns and "NAME_INCOME_TYPE" in X.columns:
            for income, occ in self._occupation_by_income.items():
                X = X.with_columns(
                    pl.when(
                        (pl.col("OCCUPATION_TYPE").is_null())
                        & (pl.col("NAME_INCOME_TYPE") == income)
                    )
                    .then(pl.lit(occ))
                    .otherwise(pl.col("OCCUPATION_TYPE"))
                    .alias("OCCUPATION_TYPE")
                )
            X = X.with_columns(
                pl.when(pl.col("OCCUPATION_TYPE").is_null())
                .then(pl.lit(self._default_occupation))
                .otherwise(pl.col("OCCUPATION_TYPE"))
                .alias("OCCUPATION_TYPE")
            )

        if self._apartment_learned_cols:
            X = X.drop(self._apartment_learned_cols)

        result_X = X.clone()

        if self._cat_modes:
            cat_exprs = []
            for col, mode in self._cat_modes.items():
                if col in result_X.columns:
                    cat_exprs.append(pl.col(col).fill_null(mode).alias(col))
            if cat_exprs:
                result_X = result_X.with_columns(cat_exprs)

        work_pdf = result_X.to_pandas()

        if self._low_missing:
            train_medians = {
                c: work_pdf[c].median() for c in self._low_missing if c in work_pdf.columns
            }
            work_pdf[self._low_missing] = work_pdf[self._low_missing].fillna(train_medians)

        if self._high_missing and self._iterative_imputer is not None:
            work_pdf[self._high_missing] = self._iterative_imputer.transform(
                work_pdf[self._high_missing]
            )

        result_X = pl.from_pandas(work_pdf).with_columns(
            [
                pl.col(col).cast(dtype)
                for col, dtype in self._int_cols.items()
                if col in result_X.columns
            ]
        )

        return result_X
