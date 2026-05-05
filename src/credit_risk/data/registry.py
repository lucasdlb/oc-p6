"""Generic registry for table-specific pipeline step implementations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator


class Registry:
    """Lazy-initialized registry of step implementations by class name.

    Each subclass defines:
        _registry: dict[str, type] = {}
        _initialized: bool = False
        _register_defaults() -> None

    Usage:
        class CleaningRegistry(Registry):
            _registry: dict[str, type] = {}
            _initialized: bool = False

            @classmethod
            def _register_defaults(cls) -> None:
                from credit_risk.data.cleaning.application import ApplicationCleaner
                cls._registry["ApplicationCleaner"] = ApplicationCleaner

        cleaner = CleaningRegistry.get("ApplicationCleaner")()
    """

    _registry: dict[str, type]
    _initialized: bool

    @classmethod
    def _register_defaults(cls) -> None:
        raise NotImplementedError

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True

    @classmethod
    def get(cls, key: str) -> type:
        cls._ensure_initialized()
        if key not in cls._registry:
            available = list(cls._registry)
            raise KeyError(f"'{key}' not found in {cls.__name__}. Available: {available}")
        return cls._registry[key]

    @classmethod
    def register(cls, key: str, impl: type) -> None:
        cls._ensure_initialized()
        cls._registry[key] = impl

    @classmethod
    @contextmanager
    def override(cls, key: str, impl: type) -> Generator[None, None, None]:
        cls._ensure_initialized()
        previous = cls._registry.get(key)
        cls._registry[key] = impl
        try:
            yield
        finally:
            if previous is not None:
                cls._registry[key] = previous
            else:
                cls._registry.pop(key, None)

    @classmethod
    def available(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._registry)
