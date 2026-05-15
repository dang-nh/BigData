"""
Portable repo / data paths — use instead of hard-coded /home/... or /data/... defaults.

Resolved order:
  1) Explicit env vars (PROJECT_ROOT optional; docker-compose typically sets *_DIR)
  2) Inside Docker: defaults under /data/... when /.dockerenv exists
  3) Locally: directories under repository root (<repo>/data/...)

Relative values in env are anchored to PROJECT_ROOT (or inferred repo root).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "inferred_repo_root",
    "mimic_data_dir",
    "parquet_output_dir",
    "results_dir",
    "schema_cypher_path",
    "checkpoints_notes_stream_dir",
]


def _in_container() -> bool:
    try:
        return Path("/.dockerenv").exists()
    except OSError:
        return False


def inferred_repo_root() -> Path:
    """Parent of ``src`` (directory containing kg_paths.py)."""
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Repository root override for locating resources (schema, notebooks, etc.)."""
    raw = os.environ.get("PROJECT_ROOT") or os.environ.get("KG_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return inferred_repo_root()


def _resolved_path(env_var: str, docker_default: Path, local_relative: Path) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = repo_root() / p
        return p.resolve()
    if _in_container():
        return docker_default.resolve()
    return (repo_root() / local_relative).resolve()


def mimic_data_dir() -> Path:
    return _resolved_path(
        "MIMIC_DATA_DIR",
        Path("/data/mimic-iii"),
        Path("data/mimic-iii"),
    )


def parquet_output_dir() -> Path:
    return _resolved_path(
        "PARQUET_OUTPUT_DIR",
        Path("/data/parquet"),
        Path("data/parquet"),
    )


def results_dir() -> Path:
    return _resolved_path(
        "RESULTS_DIR",
        Path("/data/results"),
        Path("data/results"),
    )


def schema_cypher_path() -> Path:
    raw = os.environ.get("NEO4J_SCHEMA_PATH")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = repo_root() / p
        return p.resolve()
    return (repo_root() / "src" / "graph" / "schema.cypher").resolve()


def checkpoints_notes_stream_dir() -> Path:
    """Spark Structured Streaming checkpoint for Kafka→Mongo ingest."""
    raw = os.environ.get("SPARK_CHECKPOINT_DIR")
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = repo_root() / p
        return p.resolve()
    if _in_container():
        return Path("/data/checkpoints/notes-stream").resolve()
    return (repo_root() / "data" / "checkpoints" / "notes-stream").resolve()
