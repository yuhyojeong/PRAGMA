"""Load and validate shared PRAGMA construction settings."""

from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"


def _require_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml: {key!r} must be a mapping")
    return value


def _require_string(section: dict, section_name: str, key: str) -> None:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"config.yaml: {section_name}.{key} must be a non-empty string"
        )


def _require_positive_int(section: dict, section_name: str, key: str) -> None:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"config.yaml: {section_name}.{key} must be a positive integer"
        )


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Required configuration file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a top-level mapping")

    models = _require_mapping(config, "models")
    for key in (
        "schema",
        "event_generation",
        "event_query",
        "query_rewrite",
        "trajectory_generation",
        "trajectory_query",
        "evidence_session",
        "filler_topic",
        "filler_session",
    ):
        _require_string(models, "models", key)

    concurrency = _require_mapping(config, "concurrency")
    for key in ("evidence_requests", "filler_requests"):
        _require_positive_int(concurrency, "concurrency", key)

    generation = _require_mapping(config, "generation")
    _require_positive_int(generation, "generation", "max_attempts")

    sampling = _require_mapping(config, "sampling")
    _require_positive_int(sampling, "sampling", "num_users")
    seed = sampling.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("config.yaml: sampling.seed must be an integer")

    return config


CONFIG = load_config()
