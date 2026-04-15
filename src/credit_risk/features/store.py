"""Feature store for persisting and reloading named feature sets."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


class FeatureStore:
    """Persist and reload named feature sets."""

    def __init__(self, root: str | Path = "artifacts/features"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        features: list[str],
        meta: dict | None = None,
    ) -> Path:
        """Save a feature list with optional metadata."""
        record = {
            "name": name,
            "features": features,
            "n_features": len(features),
            "saved_at": datetime.now().isoformat(),
            "meta": meta or {},
        }
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(record, indent=2))
        logger.info(f"Saved {len(features)} features to {name}")
        return path

    def load_tables(
        self,
        tables: list[str],
        suffix: str = "",
        with_records: bool = False,
    ) -> tuple[list[str], dict[str, list[str] | dict]]:
        """Load features for multiple tables.

        Args:
            tables: List of table names (e.g., ["application", "bureau"])
            suffix: Suffix to add (e.g., "_prod", "_debug")
            with_records: If True, load full record (with meta) instead of just features

        Returns:
            Tuple of (all_features, loaded_by_table)
            - all_features: Combined list of all feature names (deduplicated)
            - loaded_by_table: Dict mapping table name to features list or full record
        """
        all_features = []
        loaded_by_table = {}

        for table in tables:
            feature_name = f"{table}{suffix}"
            path = self.root / f"{feature_name}.json"
            if path.exists():
                record = json.loads(path.read_text())
                features = record["features"]
                all_features.extend(features)
                loaded_by_table[table] = record if with_records else features
                logger.info(f"Loaded {len(features)} features from {feature_name}")
            else:
                logger.warning(f"No saved features for {feature_name}")

        all_features = list(set(all_features))
        logger.info(
            f"Total: loaded {len(all_features)} unique features from {len(loaded_by_table)} tables"
        )
        return all_features, loaded_by_table
