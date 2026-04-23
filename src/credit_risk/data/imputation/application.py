"""Application table imputer."""

from __future__ import annotations

import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from polars import DataFrame


from credit_risk.data.imputation.base import TableImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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


def group_rare_categories(pdf: pd.DataFrame, min_freq: float = 0.01) -> pd.DataFrame:
    for col in pdf.select_dtypes(include="category").columns:
        freq = pdf[col].value_counts(normalize=True)
        rare = freq[freq < min_freq].index
        if len(rare) > 0:
            # 1. add "Other" to known categories first
            if "Other" not in pdf[col].cat.categories:
                pdf[col] = pdf[col].cat.add_categories("Other")
            # 2. replace rare → "Other"
            pdf[col] = pdf[col].where(~pdf[col].isin(rare), other="Other")
            # 3. now safe to remove rare categories from the registry
            pdf[col] = pdf[col].cat.remove_categories(rare)
    return pdf


class ApplicationImputer(TableImputer):
    """Imputer for application_train/test tables.

    Domain-aware imputation:
    - OWN_CAR_AGE: -1 when FLAG_OWN_CAR = "N", median when "Y"
    - AMT_REQ_CREDIT_BUREAU_*: 0 when all co-missing
    - Apartment features: 0 when HOUSETYPE_MODE is NaN, median by housing type otherwise
    - EXT_SOURCE_*: median + binary missingness flag
    - OCCUPATION_TYPE: mode by NAME_INCOME_TYPE
    """

    def impute(self, df: DataFrame) -> DataFrame:
        df = self._impute_own_car_age(df)
        df = self._impute_amt_req_credit_bureau(df)
        df = self._impute_apartment_features(df)
        df = self._impute_ext_source(df)
        df = self._impute_occupation_type(df)

        df = df.drop(APARTMENT_COLS)

        int_cols = {
            col: dtype for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        }

        pdf = df.to_pandas()

        # separate numeric and categorical
        num_cols = pdf.select_dtypes(include="number").columns.tolist()
        cat_cols = pdf.select_dtypes(include="object").columns.tolist()

        # simple fill for categoricals — mode per column
        for col in cat_cols:
            pdf[col] = pdf[col].fillna(pdf[col].mode()[0])

        # faster: limit which columns get iterative treatment
        high_missing = [c for c in num_cols if df[c].null_count() / len(df) > 0.05]
        low_missing = [c for c in num_cols if df[c].null_count() / len(df) <= 0.05]

        # cheap median fill for low-missing columns
        pdf[low_missing] = pdf[low_missing].fillna(pdf[low_missing].median())

        # expensive iterative imputation only where it matters
        imp = IterativeImputer(estimator=LGBMRegressor(n_estimators=50, verbosity=-1), max_iter=3)
        pdf[high_missing] = imp.fit_transform(pdf[high_missing])

        # iterative imputation on numerics only
        imp = IterativeImputer(
            estimator=LGBMRegressor(n_estimators=50, verbosity=-1),
            max_iter=5,
            random_state=42,
        )
        pdf[num_cols] = imp.fit_transform(pdf[num_cols])

        df = pl.from_pandas(pdf).with_columns([
            pl.col(col).cast(dtype)
            for col, dtype in int_cols.items()
        ])
        return df

    def _impute_own_car_age(self, df: DataFrame) -> DataFrame:
        if "OWN_CAR_AGE" not in df.columns or "FLAG_OWN_CAR" not in df.columns:
            return df

        median_car_age = (
            df.filter(pl.col("FLAG_OWN_CAR") == "Y").select(pl.col("OWN_CAR_AGE").median()).item()
        )
        if median_car_age is None:
            median_car_age = 0.0

        df = df.with_columns(
            pl.when(pl.col("FLAG_OWN_CAR") == "N")
            .then(pl.lit(-1.0))
            .when(pl.col("OWN_CAR_AGE").is_null())
            .then(pl.lit(median_car_age))
            .otherwise(pl.col("OWN_CAR_AGE"))
            .alias("OWN_CAR_AGE")
        )

        return df

    def _impute_amt_req_credit_bureau(self, df: DataFrame) -> DataFrame:
        cols = [c for c in AMT_REQ_CREDIT_BUREAU_COLS if c in df.columns]
        if not cols:
            return df

        all_null_expr = pl.col(cols[0]).is_null()
        for col in cols[1:]:
            all_null_expr = all_null_expr & pl.col(col).is_null()

        fill_exprs = [pl.col(c).fill_null(-1).alias(c) for c in cols]
        df = df.with_columns(*fill_exprs, all_null_expr.alias("AMT_REQ_CREDIT_BUREAU_missing"))

        return df

    def _impute_apartment_features(self, df: DataFrame) -> DataFrame:
        apartment_cols = [c for c in APARTMENT_COLS if c in df.columns]
        if not apartment_cols:
            return df

        housing_types = ["block of flats", "specific housing", "terraced house"]

        for col in apartment_cols:
            global_median = df.select(pl.col(col).median()).item() or 0.0

            median_by_housing = {}
            for ht in housing_types:
                ht_median = (
                    df.filter(pl.col("HOUSETYPE_MODE") == ht).select(pl.col(col).median()).item()
                )
                if ht_median is not None:
                    median_by_housing[ht] = ht_median

            if not median_by_housing:
                df = df.with_columns(pl.col(col).fill_null(global_median).alias(col))
                continue

            expr = pl.when(pl.col("HOUSETYPE_MODE").is_null()).then(pl.lit(0.0))
            for ht, med in median_by_housing.items():
                expr = expr.when(pl.col("HOUSETYPE_MODE") == ht).then(pl.lit(med))
            expr = expr.otherwise(pl.col(col).fill_null(global_median))

            df = df.with_columns(expr.alias(col))

        return df

    def _impute_ext_source(self, df: DataFrame) -> DataFrame:
        cols = [c for c in EXT_SOURCE_COLS if c in df.columns]
        if not cols:
            return df

        for col in cols:
            median_val = df.select(pl.col(col).median()).item() or 0.0

            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(f"{col}_missing"),
                pl.col(col).fill_null(median_val).alias(col),
            )

        return df

    def _impute_occupation_type(self, df: DataFrame) -> DataFrame:
        if "OCCUPATION_TYPE" not in df.columns or "NAME_INCOME_TYPE" not in df.columns:
            return df

        income_occ_map: dict[str, str] = {}
        for income in df["NAME_INCOME_TYPE"].unique().to_list():
            if income is None:
                continue
            filtered = df.filter(
                (pl.col("NAME_INCOME_TYPE") == income) & (pl.col("OCCUPATION_TYPE").is_not_null())
            )
            if filtered.height > 0:
                mode_occ = (
                    filtered.select(pl.col("OCCUPATION_TYPE")).to_pandas()["OCCUPATION_TYPE"].mode()
                )
                if len(mode_occ) > 0:
                    income_occ_map[income] = mode_occ[0]

        global_mode = df.select(pl.col("OCCUPATION_TYPE")).to_pandas()["OCCUPATION_TYPE"].mode()
        default_occupation = global_mode.iloc[0] if len(global_mode) > 0 else "Laborers"

        for income, occ in income_occ_map.items():
            df = df.with_columns(
                pl.when(
                    (pl.col("OCCUPATION_TYPE").is_null()) & (pl.col("NAME_INCOME_TYPE") == income)
                )
                .then(pl.lit(occ))
                .otherwise(pl.col("OCCUPATION_TYPE"))
                .alias("OCCUPATION_TYPE")
            )

        df = df.with_columns(
            pl.when(pl.col("OCCUPATION_TYPE").is_null())
            .then(pl.lit(default_occupation))
            .otherwise(pl.col("OCCUPATION_TYPE"))
            .alias("OCCUPATION_TYPE")
        )

        return df
