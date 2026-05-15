#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DVV_RUNNER = ROOT / "src" / "01-dvv-calculation" / "runner.py"


def load_dvv_runner():
    spec = importlib.util.spec_from_file_location("dvv_calculation_runner", DVV_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dv/v runner from {DVV_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrated Iceland dv/v and pressure workflow.")
    parser.add_argument("--config", default="config.toml", help="Path to workflow config.")
    parser.add_argument(
        "--stage",
        default=None,
        choices=["dvv_calculation"],
        help="Workflow stage to run. Defaults to workflow.active_stage in config.",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=["msnoise", "mcmc", "both"],
        help="Override dvv_calculation.method from config.",
    )
    args = parser.parse_args()

    runner = load_dvv_runner()
    cfg = runner.load_config(args.config)
    stage = args.stage or cfg.get("workflow", {}).get("active_stage", "dvv_calculation")

    if stage != "dvv_calculation":
        raise ValueError(f"Unsupported stage for now: {stage}")

    result = runner.run_dvv_calculation(cfg, method_override=args.method)
    runner.print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
