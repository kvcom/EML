from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OptimisationConfig(BaseModel):
    trials: int = 500
    holdout_fraction: float = 0.2
    timeout_seconds: int | None = None


class AppConfig(BaseModel):
    database_path: str = "data/db/euromillions.sqlite"
    min_training_draws: int = 200
    top_predictions: int = 3
    random_seed: int = 42
    main_candidate_pool_size: int = 10_000
    star_candidate_pool_size: int = 66
    enabled_sources: list[str] = Field(default_factory=list)
    optimisation: OptimisationConfig = Field(default_factory=OptimisationConfig)

    def database_file(self) -> Path:
        return Path(self.database_path)


def load_config(path: str = "config/default.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return AppConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must map keys to values")
    return AppConfig.model_validate(raw)


def merge_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    merged = config.model_dump()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return AppConfig.model_validate(merged)
