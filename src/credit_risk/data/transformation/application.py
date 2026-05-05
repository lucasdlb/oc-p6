"""Transformer for application table features.

Feature engineering strategy based on top Kaggle solutions:
- EXT_SOURCE interactions (highest importance features in the dataset)
- Employment stress and stability signals
- Document anomaly flags
- Financial stress ratios beyond simple credit/income
- Age-based risk interactions
- Flag aggregations (social, contact, region)
- Loan purpose / contract type interactions
"""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class ApplicationTransformer(StatelessStep):
    """Transformer for application table features.

    Feature groups, ordered by expected impact:
    1. EXT_SOURCE combinations       — highest LGBM importance in competition
    2. Employment stress signals      — strong default predictors
    3. Financial ratio enrichment     — beyond basic credit/income
    4. Document & flag anomalies      — behavioural risk signals
    5. Age × risk interactions        — non-linear age effects
    6. Goods / loan structure         — repayment burden shape
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        cols = set(X.columns)
        new_cols = []

        # ------------------------------------------------------------------ #
        # 1. EXT_SOURCE — these three bureau-like scores are the most         #
        #    predictive single features; their combinations matter even more  #
        # ------------------------------------------------------------------ #
        has_ext = {
            1: "EXT_SOURCE_1" in cols,
            2: "EXT_SOURCE_2" in cols,
            3: "EXT_SOURCE_3" in cols,
        }

        if has_ext[1] and has_ext[2] and has_ext[3]:
            new_cols += [
                # Simple mean — top-5 feature in most published solutions
                (
                    (pl.col("EXT_SOURCE_1") + pl.col("EXT_SOURCE_2") + pl.col("EXT_SOURCE_3")) / 3
                ).alias("ext_source_mean"),
                # Weighted mean (source 2 & 3 tend to be stronger)
                (
                    pl.col("EXT_SOURCE_1") * 0.25
                    + pl.col("EXT_SOURCE_2") * 0.375
                    + pl.col("EXT_SOURCE_3") * 0.375
                ).alias("ext_source_weighted_mean"),
                # Spread: high variance in scores = instability signal
                (
                    (
                        (pl.col("EXT_SOURCE_1") - pl.col("EXT_SOURCE_2")).pow(2)
                        + (pl.col("EXT_SOURCE_2") - pl.col("EXT_SOURCE_3")).pow(2)
                        + (pl.col("EXT_SOURCE_1") - pl.col("EXT_SOURCE_3")).pow(2)
                    )
                    / 3
                )
                .sqrt()
                .alias("ext_source_std"),
                # Product — captures joint low-score risk
                (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3")).alias(
                    "ext_source_product"
                ),
                # Min score — worst external rating
                pl.min_horizontal("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3").alias(
                    "ext_source_min"
                ),
            ]

        if has_ext[2] and has_ext[3]:
            new_cols.append((pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3")).alias("ext_2_x_3"))

        # EXT_SOURCE vs credit amount: good score but large loan = stressed
        if has_ext[2] and "AMT_CREDIT" in cols:
            new_cols.append(
                (pl.col("AMT_CREDIT") / (pl.col("EXT_SOURCE_2") + 1e-5)).alias("credit_per_ext2")
            )

        if has_ext[3] and "AMT_CREDIT" in cols:
            new_cols.append(
                (pl.col("AMT_CREDIT") / (pl.col("EXT_SOURCE_3") + 1e-5)).alias("credit_per_ext3")
            )

        # ------------------------------------------------------------------ #
        # 2. Employment stress signals                                         #
        # ------------------------------------------------------------------ #

        if "DAYS_BIRTH" in cols:
            new_cols.append((pl.col("DAYS_BIRTH") / -365.25).alias("AGE"))

        if "DAYS_BIRTH" in cols and "YEARS_EMPLOYED" in cols:
            # Fraction of life spent employed — stability proxy
            new_cols.append(
                (pl.col("YEARS_EMPLOYED").abs() / (pl.col("DAYS_BIRTH").abs() + 1)).alias(
                    "employed_to_age_ratio"
                )
            )

        if "YEARS_EMPLOYED" in cols and "AMT_INCOME_TOTAL" in cols:
            # Income per year of employment — high income but short tenure = risk
            new_cols.append(
                (pl.col("AMT_INCOME_TOTAL") / (pl.col("YEARS_EMPLOYED").abs() / 365.25 + 1)).alias(
                    "income_per_employed_year"
                )
            )

        if "DAYS_REGISTRATION" in cols and "DAYS_BIRTH" in cols:
            # How old was the client when they registered? Late registration = instability
            new_cols.append(
                ((pl.col("DAYS_BIRTH") - pl.col("DAYS_REGISTRATION")).abs() / 365.25).alias(
                    "age_at_registration"
                )
            )

        if "DAYS_ID_PUBLISH" in cols and "DAYS_BIRTH" in cols:
            # Recently renewed ID relative to age = possibly hiding history
            new_cols.append(
                (pl.col("DAYS_ID_PUBLISH").abs() / (pl.col("DAYS_BIRTH").abs() + 1)).alias(
                    "id_renewal_rate"
                )
            )

        # ------------------------------------------------------------------ #
        # 3. Financial ratio enrichment                                        #
        # ------------------------------------------------------------------ #
        if "AMT_INCOME_TOTAL" in cols and "AMT_CREDIT" in cols:
            new_cols.append(
                (pl.col("AMT_CREDIT") / (pl.col("AMT_INCOME_TOTAL") + 1)).alias(
                    "credit_to_income_ratio"
                )
            )

        if "AMT_INCOME_TOTAL" in cols and "AMT_ANNUITY" in cols:
            new_cols += [
                (pl.col("AMT_ANNUITY") / (pl.col("AMT_INCOME_TOTAL") + 1)).alias(
                    "annuity_to_income_ratio"
                ),
                # Disposable income after annuity payments
                (pl.col("AMT_INCOME_TOTAL") - pl.col("AMT_ANNUITY")).alias("disposable_income"),
                # Monthly income vs monthly annuity stress
                (pl.col("AMT_ANNUITY") / (pl.col("AMT_INCOME_TOTAL") / 12 + 1)).alias(
                    "annuity_to_monthly_income"
                ),
            ]

        if "AMT_CREDIT" in cols and "AMT_ANNUITY" in cols:
            new_cols.append(
                (pl.col("AMT_CREDIT") / (pl.col("AMT_ANNUITY") + 1)).alias(
                    "loan_term_proxy"  # longer implied term = more risk
                )
            )

        if "AMT_GOODS_PRICE" in cols and "AMT_CREDIT" in cols:
            new_cols += [
                (pl.col("AMT_CREDIT") - pl.col("AMT_GOODS_PRICE")).alias(
                    "credit_minus_goods"  # positive = borrowing more than goods worth
                ),
                (pl.col("AMT_CREDIT") / (pl.col("AMT_GOODS_PRICE") + 1)).alias(
                    "credit_to_goods_ratio"  # LTV proxy
                ),
                (
                    (pl.col("AMT_CREDIT") - pl.col("AMT_GOODS_PRICE"))
                    / (pl.col("AMT_GOODS_PRICE") + 1)
                ).alias("ltv_excess_ratio"),
            ]

        if "AMT_INCOME_TOTAL" in cols and "CNT_FAM_MEMBERS" in cols:
            new_cols.append(
                (pl.col("AMT_INCOME_TOTAL") / (pl.col("CNT_FAM_MEMBERS") + 1)).alias(
                    "income_per_family_member"
                )
            )

        if "AMT_INCOME_TOTAL" in cols and "CNT_CHILDREN" in cols:
            new_cols.append(
                (pl.col("CNT_CHILDREN") / (pl.col("AMT_INCOME_TOTAL") / 10_000 + 1)).alias(
                    "children_per_10k_income"  # dependency burden
                )
            )

        # ------------------------------------------------------------------ #
        # 4. Document & flag anomaly aggregations                              #
        # ------------------------------------------------------------------ #
        doc_cols = [c for c in cols if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            new_cols += [
                # Total documents submitted
                pl.sum_horizontal(*[pl.col(c) for c in doc_cols]).alias(
                    "total_documents_submitted"
                ),
                # Missing document rate — not submitting docs is a red flag
                (
                    pl.sum_horizontal(*[pl.col(c).eq(0).cast(pl.Int32) for c in doc_cols])
                    / len(doc_cols)
                ).alias("document_missing_rate"),
            ]

        # Social circle default flags (OBS = observed, DEF = defaulted)
        social_obs_cols = [c for c in cols if "OBS_30" in c or "OBS_60" in c]
        social_def_cols = [c for c in cols if "DEF_30" in c or "DEF_60" in c]
        if social_def_cols and social_obs_cols:
            new_cols.append(
                pl.sum_horizontal(*[pl.col(c) for c in social_def_cols]).alias(
                    "social_circle_defaults"
                )
            )

        # Contact / reachability flags
        contact_cols = [c for c in cols if c.startswith("FLAG_") and "CONTACT" in c]
        if contact_cols:
            new_cols.append(
                pl.sum_horizontal(*[pl.col(c) for c in contact_cols]).alias("total_contact_flags")
            )

        # Region risk — high population relative = urban, different risk profile
        if "REGION_POPULATION_RELATIVE" in cols and "AMT_INCOME_TOTAL" in cols:
            new_cols.append(
                (pl.col("AMT_INCOME_TOTAL") * pl.col("REGION_POPULATION_RELATIVE")).alias(
                    "income_x_region_pop"
                )
            )

        if "REGION_RATING_CLIENT" in cols and "REGION_RATING_CLIENT_W_CITY" in cols:
            # Disagreement between region ratings = inconsistency signal
            new_cols.append(
                (pl.col("REGION_RATING_CLIENT") - pl.col("REGION_RATING_CLIENT_W_CITY")).alias(
                    "region_rating_mismatch"
                )
            )

        # ------------------------------------------------------------------ #
        # 5. Age × risk interactions                                           #
        # ------------------------------------------------------------------ #
        if "DAYS_BIRTH" in cols:
            age_years = pl.col("DAYS_BIRTH") / -365.25

            if "AMT_INCOME_TOTAL" in cols:
                new_cols.append(
                    (pl.col("AMT_INCOME_TOTAL") / (age_years + 1)).alias(
                        "income_per_age_year"  # young high earners vs old low earners
                    )
                )

            if has_ext[2]:
                new_cols.append(
                    (pl.col("EXT_SOURCE_2") / (age_years + 1)).alias(
                        "ext2_per_age"  # external score relative to age
                    )
                )

            if "AMT_CREDIT" in cols:
                new_cols.append(
                    (pl.col("AMT_CREDIT") / (age_years + 1)).alias("credit_per_age_year")
                )

        # ------------------------------------------------------------------ #
        # 6. Car / real estate wealth signals                                  #
        # ------------------------------------------------------------------ #
        if "FLAG_OWN_CAR" in cols and "AMT_INCOME_TOTAL" in cols:
            new_cols.append(
                (
                    pl.when(pl.col("FLAG_OWN_CAR") == "Y").then(1).otherwise(0).cast(pl.Float64)
                    * pl.col("AMT_INCOME_TOTAL")
                ).alias("car_owner_income")
            )

        if "FLAG_OWN_REALTY" in cols and "AMT_CREDIT" in cols:
            new_cols.append(
                pl.when(pl.col("FLAG_OWN_REALTY") == "Y")
                .then(pl.col("AMT_CREDIT"))
                .otherwise(pl.col("AMT_CREDIT") * 1.2)  # no realty = higher risk weight
                .alias("credit_realty_adjusted")
            )

        # OWN_CAR_AGE: older car + large loan = stretched finances
        if "OWN_CAR_AGE" in cols and "AMT_CREDIT" in cols:
            new_cols.append(
                (pl.col("OWN_CAR_AGE") * pl.col("AMT_CREDIT")).alias("car_age_x_credit")
            )

        # ------------------------------------------------------------------ #
        # Apply all new columns at once                                        #
        # ------------------------------------------------------------------ #
        if new_cols:
            X = X.with_columns(new_cols)

        _DERIVE_AND_DROP: dict[str, str] = {
            "DAYS_BIRTH": "AGE",
            "DAYS_REGISTRATION": "age_at_registration",
            "DAYS_ID_PUBLISH": "id_renewal_rate",
        }

        X = X.drop(
            [
                raw
                for raw, derived in _DERIVE_AND_DROP.items()
                if raw in X.columns and derived in X.columns  # only drop if derivation succeeded
            ]
        )

        return X
