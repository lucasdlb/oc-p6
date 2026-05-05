#!/usr/bin/env python
"""Processing configuration grid search — per-table isolation.

Sweeps the axes defined in sweep_processing.toml.  For each table:
  - One parent MLflow run logged to "sweep_processing_{mode}"
  - Each combo is a nested child run inside the table parent
  - The table parent logs: sweep axes, best metric, best config artifact
  - Each child run logs: full processing config params + cv_roc_auc + std + n_features

After the sweep, update data.toml with the best config per table and proceed
to rfe_cv.py.

Usage:
    RUN_MODE=debug uv run python scripts/sweep_processing.py
    RUN_MODE=dev   uv run python scripts/sweep_processing.py
    uv run python scripts/sweep_processing.py   # defaults to prod
"""

from __future__ import annotations

import copy
import itertools
import logging
import tomllib
import warnings
from dataclasses import dataclass
from typing import Any

import mlflow

from credit_risk.config import load_config
from credit_risk.config.config import CONFIG_DIR
from credit_risk.config.models import Config
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.pipeline.evaluator import run_cv

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single processing configuration run."""

    combo: dict[str, Any]
    cv_roc_auc: float
    cv_roc_auc_std: float
    n_features: int


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_get(d: dict, path: str) -> Any | None:
    keys = path.split(".")
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _deep_set(d: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _extract_axes(
    overlay: dict,
    base: dict,
    prefix: str = "",
) -> tuple[dict[str, list], dict[str, Any]]:
    """Recursively split overlay into sweep axes (lists) and fixed overrides (scalars)."""
    axes: dict[str, list] = {}
    overrides: dict[str, Any] = {}

    for key, value in overlay.items():
        full_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            base_sub = base.get(key, {}) if isinstance(base, dict) else {}
            sub_axes, sub_overrides = _extract_axes(value, base_sub, full_path)
            axes.update(sub_axes)
            overrides.update(sub_overrides)
        elif isinstance(value, list) and value:
            axes[full_path] = value
        elif value is not None:
            overrides[full_path] = value

    return axes, overrides


def _build_config(base_cfg: Config, combination: dict[str, Any]) -> Config:
    cfg_dict = copy.deepcopy(base_cfg.model_dump())
    if any(path.startswith("model.params.") for path in combination):
        _deep_get(cfg_dict, "model").pop("params", None)
        _deep_set(cfg_dict, "model.params", {})
    for path, value in combination.items():
        _deep_set(cfg_dict, path, value)
    return Config.model_validate(cfg_dict)


def _short_key(path: str) -> str:
    """data.application.encoder → encoder."""
    return path.rsplit(".", 1)[-1]


def _combo_label(combo: dict[str, Any]) -> str:
    """Short human-readable label — strip table prefix for brevity."""
    return ", ".join(f"{_short_key(k)}={v}" for k, v in sorted(combo.items()))


# ---------------------------------------------------------------------------
# Per-table sweep
# ---------------------------------------------------------------------------


def _sweep_table(
    table: str,
    table_axes: dict[str, list],
    table_overrides: dict[str, Any],
    base_cfg: Config,
    run_mode: str,
    ml_logger: MlflowLogger,
) -> list[RunResult]:
    """Run all combos for one table; each combo nested inside a table parent run.

    The table parent run logs:
      - swept axis names and their candidate values (comma-sep strings)
      - best cv_roc_auc + best config as an artifact

    Each nested combo run logs:
      - full processing config (cleaner, imputer, aggregator, transformer, encoder)
      - swept param values
      - cv_roc_auc, cv_roc_auc_std, n_features
    """
    axis_names = sorted(table_axes.keys())
    axis_values = [table_axes[k] for k in axis_names]
    n_runs = 1
    for vals in axis_values:
        n_runs *= len(vals)

    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Sweeping: {table}  ({n_runs} runs)")
    for name, vals in sorted(table_axes.items()):
        logger.info(f"    {_short_key(name)}: {vals}")
    logger.info(f"{'=' * 60}")

    results: list[RunResult] = []

    # ── Table parent run ──────────────────────────────────────────────────
    with ml_logger.start_run(run_name=f"{run_mode}__{table}"):
        # Log the swept axes as comma-separated params so they're searchable
        for axis_path, vals in sorted(table_axes.items()):
            ml_logger.log_param(f"sweep_{_short_key(axis_path)}", ",".join(str(v) for v in vals))
        ml_logger.log_params({"table": table, "run_mode": run_mode, "n_combos": n_runs})

        # ── One nested run per combo ──────────────────────────────────────
        for combo_values in itertools.product(*axis_values):
            combo = {**table_overrides, **dict(zip(axis_names, combo_values, strict=True))}
            cfg = _build_config(base_cfg, combo)
            label = _combo_label(combo)

            logger.info(f"  [{label}]")

            with ml_logger.start_run(run_name=f"{table}__{label}"):
                # Full processing config for this combo
                table_cfg = getattr(cfg.data, table)
                ml_logger.log_params(
                    {
                        "table": table,
                        "cleaner": table_cfg.cleaner,
                        "imputer": table_cfg.imputer,
                        "aggregator": table_cfg.aggregator,
                        "transformer": table_cfg.transformer,
                        "encoder": table_cfg.encoder,
                    }
                )
                # Swept values explicitly
                for k, v in combo.items():
                    ml_logger.log_param(_short_key(k), v)

                auc_mean, cv_std, n_features = run_cv(cfg=cfg, tables=[table])

                ml_logger.log_metrics(
                    {
                        "cv_roc_auc": auc_mean,
                        "cv_roc_auc_std": cv_std,
                        "n_features": n_features,
                    }
                )

            result = RunResult(
                combo=combo,
                cv_roc_auc=auc_mean,
                cv_roc_auc_std=cv_std,
                n_features=n_features,
            )
            results.append(result)
            logger.info(f"    → ROC AUC: {auc_mean:.4f} ± {cv_std:.4f}  ({n_features} features)")

        # ── Log best result on the table parent run ───────────────────────
        if results:
            best = max(results, key=lambda r: r.cv_roc_auc)
            ml_logger.log_metrics(
                {
                    "best_cv_roc_auc": best.cv_roc_auc,
                    "best_cv_roc_auc_std": best.cv_roc_auc_std,
                    "best_n_features": best.n_features,
                }
            )

            best_config = {
                "table": table,
                "cv_roc_auc": best.cv_roc_auc,
                "cv_roc_auc_std": best.cv_roc_auc_std,
                "n_features": best.n_features,
                "config": {_short_key(k): v for k, v in best.combo.items()},
                "all_results": [
                    {
                        "config": {_short_key(k): v for k, v in r.combo.items()},
                        "cv_roc_auc": r.cv_roc_auc,
                        "cv_roc_auc_std": r.cv_roc_auc_std,
                        "n_features": r.n_features,
                    }
                    for r in sorted(results, key=lambda r: -r.cv_roc_auc)
                ],
            }
            ml_logger.log_dict_artifact(best_config, "best_config.json")

    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _log_table_summary(table: str, results: list[RunResult]) -> None:
    if not results:
        return

    sorted_results = sorted(results, key=lambda r: -r.cv_roc_auc)
    best = sorted_results[0]

    logger.info("")
    logger.info(f"{'─' * 60}")
    logger.info(f"  {table.upper()} — {len(results)} runs, best: {best.cv_roc_auc:.4f}")
    logger.info(f"{'─' * 60}")
    logger.info(f"  {'ROC AUC':>8}  {'±std':>6}  {'feats':>5}  config")
    logger.info(f"  {'─' * 8}  {'─' * 6}  {'─' * 5}  {'─' * 30}")
    for r in sorted_results:
        marker = " ◀" if r is best else ""
        logger.info(
            f"  {r.cv_roc_auc:.4f}    {r.cv_roc_auc_std:.4f}   {r.n_features:>5}  "
            f"{_combo_label(r.combo)}{marker}"
        )
    logger.info("")
    logger.info(f"  Best config for {table}:")
    for k, v in sorted(best.combo.items()):
        logger.info(f'    {_short_key(k)} = "{v}"')
    logger.info(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    base_cfg = load_config("selection", "model")
    run_mode = base_cfg.run.mode

    with open(CONFIG_DIR / "sweep_processing.toml", "rb") as f:
        sweep_toml = tomllib.load(f)

    mlflow.set_tracking_uri(base_cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment(f"sweep_processing_{run_mode}")
    ml_logger = MlflowLogger()

    # Parse all axes and overrides from the toml
    base_dict = base_cfg.model_dump()
    all_axes: dict[str, list] = {}
    all_overrides: dict[str, Any] = {}
    for section_key, section_val in sweep_toml.items():
        base_section = _deep_get(base_dict, section_key) or {}
        axes, overrides = _extract_axes(section_val, base_section, prefix=section_key)
        all_axes.update(axes)
        all_overrides.update(overrides)

    if not all_axes:
        logger.warning("No sweep axes found in sweep_processing.toml — nothing to run.")
        return

    tables_with_axes = sorted({k.split(".")[1] for k in all_axes if k.startswith("data.")})

    if not tables_with_axes:
        logger.warning("No data.* axes found — nothing to sweep.")
        return

    total_runs = sum(
        1
        for table in tables_with_axes
        for _ in itertools.product(
            *[v for k, v in all_axes.items() if k.startswith(f"data.{table}.")]
        )
    )

    logger.info(f"Mode: {run_mode}")
    logger.info(f"Tables: {tables_with_axes}")
    logger.info(f"Total runs: {total_runs}")

    all_results: dict[str, list[RunResult]] = {}

    for table in tables_with_axes:
        table_axes = {k: v for k, v in all_axes.items() if k.startswith(f"data.{table}.")}
        table_overrides = {k: v for k, v in all_overrides.items() if k.startswith(f"data.{table}.")}

        results = _sweep_table(
            table=table,
            table_axes=table_axes,
            table_overrides=table_overrides,
            base_cfg=base_cfg,
            run_mode=run_mode,
            ml_logger=ml_logger,
        )
        all_results[table] = results
        _log_table_summary(table, results)

    # Final cross-table console summary
    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info("  SWEEP COMPLETE — Best config per table:")
    logger.info(f"  {'Table':<25}  {'ROC AUC':>8}  config")
    logger.info(f"  {'─' * 25}  {'─' * 8}  {'─' * 35}")
    for table, results in all_results.items():
        if results:
            best = max(results, key=lambda r: r.cv_roc_auc)
            logger.info(f"  {table:<25}  {best.cv_roc_auc:.4f}    {_combo_label(best.combo)}")
    logger.info(f"{'=' * 60}")
    logger.info("")
    logger.info("  → Update data.toml with the best config per table, then run rfe_cv.py")


if __name__ == "__main__":
    main()
