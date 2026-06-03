from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MODEL_PATH = Path(__file__).resolve().with_name("model.py")
OUTPUT_DIR = "outputs"
OUTPUT_SUBDIR = "02-pressure-modeling"
OUTPUT_FILENAME = "pore_pressure_output.csv"
INCLUDE_SNOW_LOADING = True


@dataclass(frozen=True)
class PressureModelingResult:
    output_csv: Path
    rows: int
    columns: list[str]
    start: str
    end: str
    depths_m: tuple[float, ...]
    validation: dict[str, Any]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Only TOML config files are supported: {config_path}")

    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: dict, configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg["_config_dir"]) / path).resolve()


def require_mapping(cfg: dict, key: str) -> dict:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required config object: {key}")
    return value


def require_value(mapping: dict, key: str, context: str) -> object:
    if key not in mapping or mapping[key] in (None, ""):
        raise KeyError(f"Missing required config value: {context}.{key}")
    return mapping[key]


def require_configured(mapping: dict, key: str, context: str) -> object:
    if key not in mapping or mapping[key] is None:
        raise KeyError(f"Missing required config key: {context}.{key}")
    return mapping[key]


def require_bool(mapping: dict, key: str, context: str) -> bool:
    value = require_value(mapping, key, context)
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{key} must be a boolean")
    return value


def config_value(mapping: dict, key: str, default):
    value = mapping.get(key, default)
    if value in (None, ""):
        return default
    return value


def _load_model_module():
    module_dir = str(MODEL_PATH.parent)
    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location("pressure_modeling_model", MODEL_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pressure model from {MODEL_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(module_dir)
        except ValueError:
            pass


def _as_optional_path(cfg: dict, configured_path: object) -> Path | None:
    if configured_path in (None, ""):
        return None
    return resolve_path(cfg, str(configured_path))


def _validate_input_file(path: Path, context: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{context} must be a file: {path}")


def _pressure_time_window(cfg: dict) -> tuple[str, str]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = require_mapping(dvv_cfg, "msnoise")
    return (
        str(require_value(msnoise_cfg, "start_date", "dvv_calculation.msnoise")),
        str(require_value(msnoise_cfg, "end_date", "dvv_calculation.msnoise")),
    )


def _validate_output(cfg: dict, result: PressureModelingResult) -> dict[str, Any]:
    pressure_cfg = require_mapping(cfg, "pressure_modeling")
    if not require_bool(pressure_cfg, "validate_outputs", "pressure_modeling"):
        return {}

    min_rows = int(require_value(pressure_cfg, "min_output_rows", "pressure_modeling"))
    if not result.output_csv.exists():
        raise FileNotFoundError(f"Pressure-modeling output is missing: {result.output_csv}")
    if result.rows < min_rows:
        raise RuntimeError(
            f"Pressure-modeling output has {result.rows} rows; expected at least {min_rows}: {result.output_csv}"
        )

    required_prefixes = ["Pp_total_z", "dPp_total_z"]
    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(column.startswith(prefix) for column in result.columns)
    ]
    if missing_prefixes:
        raise RuntimeError(
            "Pressure-modeling output is missing expected modeled-pressure columns: "
            + ", ".join(missing_prefixes)
        )

    return {
        "output_csv": result.output_csv,
        "rows": result.rows,
        "columns": len(result.columns),
        "first_time": result.start,
        "last_time": result.end,
    }


def run_pressure_modeling(cfg: dict) -> PressureModelingResult:
    pressure_cfg = require_mapping(cfg, "pressure_modeling")

    if not require_bool(pressure_cfg, "enabled", "pressure_modeling"):
        raise ValueError("pressure_modeling.enabled is false; set it to true before running this stage")

    gwl_csv_path = resolve_path(cfg, require_value(pressure_cfg, "groundwater_csv_path", "pressure_modeling"))
    atmospheric_pressure_path = resolve_path(
        cfg, require_value(pressure_cfg, "atmospheric_pressure_csv_path", "pressure_modeling")
    )
    snow_path = _as_optional_path(cfg, require_configured(pressure_cfg, "snow_csv_path", "pressure_modeling"))
    external_pressure_loading_path = _as_optional_path(
        cfg, config_value(pressure_cfg, "external_pressure_loading_csv_path", "")
    )
    output_dir = resolve_path(cfg, OUTPUT_DIR)
    output_csv = output_dir / OUTPUT_SUBDIR / OUTPUT_FILENAME

    _validate_input_file(gwl_csv_path, "pressure_modeling.groundwater_csv_path")
    _validate_input_file(atmospheric_pressure_path, "pressure_modeling.atmospheric_pressure_csv_path")
    if snow_path is None:
        raise KeyError("pressure_modeling.snow_csv_path is required because snow loading is enabled")
    _validate_input_file(snow_path, "pressure_modeling.snow_csv_path")
    if external_pressure_loading_path is not None:
        _validate_input_file(
            external_pressure_loading_path,
            "pressure_modeling.external_pressure_loading_csv_path",
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    depths_m = tuple(float(depth) for depth in require_value(pressure_cfg, "depths_m", "pressure_modeling"))
    if not depths_m:
        raise ValueError("pressure_modeling.depths_m must contain at least one depth")

    start_time, end_time = _pressure_time_window(cfg)
    model = _load_model_module()
    out = model.run_pore_pressure_workflow(
        gwl_csv_path=gwl_csv_path,
        atmospheric_pressure_csv_path=atmospheric_pressure_path,
        output_csv_path=output_csv,
        snow_csv_path=snow_path,
        include_snow_loading=INCLUDE_SNOW_LOADING,
        start=start_time,
        end=end_time,
        resample_rule=str(require_value(pressure_cfg, "resample_rule", "pressure_modeling")),
        depths_m=depths_m,
        rho_w=float(require_value(pressure_cfg, "rho_w", "pressure_modeling")),
        g=float(require_value(pressure_cfg, "gravity", "pressure_modeling")),
        skempton_b=float(require_value(pressure_cfg, "skempton_b", "pressure_modeling")),
        undrained_poisson_ratio=float(
            require_value(pressure_cfg, "undrained_poisson_ratio", "pressure_modeling")
        ),
        hydraulic_diffusivity_m2_s=float(
            require_value(pressure_cfg, "hydraulic_diffusivity_m2_s", "pressure_modeling")
        ),
        poisson_ratio=float(require_value(pressure_cfg, "poisson_ratio", "pressure_modeling")),
        youngs_modulus_pa=float(require_value(pressure_cfg, "youngs_modulus_pa", "pressure_modeling")),
        m_over_mu_ratio=float(require_value(pressure_cfg, "m_over_mu_ratio", "pressure_modeling")),
        horizontal_wavenumber_m_inv=float(
            require_value(pressure_cfg, "horizontal_wavenumber_m_inv", "pressure_modeling")
        ),
        thermoelastic_spatial_factor=float(
            require_value(pressure_cfg, "thermoelastic_spatial_factor", "pressure_modeling")
        ),
        thermal_expansion_coeff_c_inv=float(
            require_value(pressure_cfg, "thermal_expansion_coeff_c_inv", "pressure_modeling")
        ),
        thermal_diffusivity_m2_s=float(
            require_value(pressure_cfg, "thermal_diffusivity_m2_s", "pressure_modeling")
        ),
        incompetent_layer_thickness_m=float(
            require_value(pressure_cfg, "incompetent_layer_thickness_m", "pressure_modeling")
        ),
        snow_density_kg_m3=float(require_value(pressure_cfg, "snow_density_kg_m3", "pressure_modeling")),
        external_pressure_loading_csv_path=external_pressure_loading_path,
    )

    result = PressureModelingResult(
        output_csv=output_csv,
        rows=int(len(out)),
        columns=[str(column) for column in out.columns],
        start=str(out.index[0]) if len(out) else "",
        end=str(out.index[-1]) if len(out) else "",
        depths_m=depths_m,
        validation={},
    )
    validation = _validate_output(cfg, result)
    return PressureModelingResult(
        output_csv=result.output_csv,
        rows=result.rows,
        columns=result.columns,
        start=result.start,
        end=result.end,
        depths_m=result.depths_m,
        validation=validation,
    )


def print_result(result: PressureModelingResult) -> None:
    print("\nPressure-modeling stage complete.")
    print(f"Output CSV: {result.output_csv}")
    print(f"Rows: {result.rows}")
    print(f"Time range: {result.start} to {result.end}")
    print(f"Depths (m): {', '.join(str(depth) for depth in result.depths_m)}")
    if result.validation:
        print("Validation:")
        for key, value in result.validation.items():
            print(f"  {key}: {value}")
