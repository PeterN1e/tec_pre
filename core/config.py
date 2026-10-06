from __future__ import annotations

import copy
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_overrides(
    config: Dict[str, Any],
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, raw_value = item.split("=", 1)
        _set_dotted(updated, key.strip(), _parse_scalar(raw_value.strip()))
    return updated


def model_slug(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_")


def _normalize_data_root(config: Dict[str, Any]) -> None:
    """Auto-convert Windows drive paths to WSL mount points on Linux."""
    data_root = config.get("data", {}).get("data_root", "")
    if platform.system() == "Linux" and re.match(r"^[A-Za-z]:[/\\]", str(data_root)):
        drive = str(data_root)[0].lower()
        rest = re.sub(r"^[A-Za-z]:[/\\]*", "", str(data_root))
        config["data"]["data_root"] = f"/mnt/{drive}/{rest}"


def load_config(
    model_name: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_DIR / "base.yaml"
    config = _read_yaml(config_path)

    name = model_name or config.get("model", {}).get("name")
    if not name:
        raise ValueError("model_name is required when the base config has no model.name")

    model_path = DEFAULT_CONFIG_DIR / "models" / f"{model_slug(name)}.yaml"
    if model_path.exists():
        config = deep_merge(config, _read_yaml(model_path))

    config.setdefault("model", {})
    config["model"]["name"] = name
    config["_meta"] = {
        "base_config": str(config_path.resolve()),
        "model_config": str(model_path.resolve()) if model_path.exists() else None,
    }

    data_root = os.getenv("TEC_DATA_ROOT")
    if data_root:
        config.setdefault("data", {})["data_root"] = data_root

    _normalize_data_root(config)

    return apply_overrides(config, overrides)


def resolve_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"]).expanduser()
    return {
        "data_root": str(data_root),
        "tec_dir": str(data_root / data_cfg["tec_subdir"]),
        "indices_dir": str(data_root / data_cfg["indices_subdir"]),
    }


def resolve_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", "save")).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def save_config(config: Dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return target

