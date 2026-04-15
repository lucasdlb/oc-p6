"""Feature store for persisting and reloading named feature sets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


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
        return path

    def load(self, name: str) -> list[str]:
        """Load a feature list by name."""
        path = self.root / f"{name}.json"
        if not path.exists():
            available = [p.stem for p in self.root.glob("*.json")]
            raise KeyError(f"No feature set '{name}'. Available: {available}")
        return json.loads(path.read_text())["features"]

    def load_or_none(self, name: str) -> list[str] | None:
        """Load feature list, return None if not found."""
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())["features"]

    def load_or_empty(self, name: str) -> list[str]:
        """Load feature list, return empty list if not found."""
        result = self.load_or_none(name)
        return result if result else []

    def load_record(self, name: str) -> dict:
        """Load full record including metadata."""
        path = self.root / f"{name}.json"
        return json.loads(path.read_text())

    def available(self) -> list[str]:
        return [p.stem for p in self.root.glob("*.json")]
