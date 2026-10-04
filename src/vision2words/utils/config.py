"""YAML config loading with dotted-key CLI overrides.

Every training/evaluation run is driven by a YAML file under configs/, and the
resolved config (including any --set overrides) is written next to the
checkpoint it produced, so a run can always be reproduced or at least
inspected after the fact.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` strings (as given to ``--set``) to a config.

    Values are parsed with ``yaml.safe_load`` so ``32``, ``1e-3`` and ``true``
    come back as the right Python type instead of a string.
    """
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key.strip(), yaml.safe_load(raw_value))
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = load_yaml(path)
    return apply_overrides(config, overrides)
