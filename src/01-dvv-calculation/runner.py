from __future__ import annotations

import csv
import math
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class ObservationSystem:
    dates: list[str]
    observations: np.ndarray
    g: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class DvvRunResult:
    method: str
    msnoise_commands: list[list[str]]
    mcmc_outputs: dict[str, Path]
    validation: dict[str, object]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Only TOML config files are supported: {config_path}")

    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: dict, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
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


def _first_present(row: dict[str, str], candidates: Iterable[str]) -> str | None:
    normalized = {key.lower(): key for key in row}
    for candidate in candidates:
        if not candidate:
            continue
        key = normalized.get(candidate.lower())
        if key is not None and row.get(key, "") != "":
            return row[key]
    return None


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def _parse_utc_date(value: str):
    from obspy import UTCDateTime

    return UTCDateTime(value)


def _iter_days(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    current = start
    while current < end:
        next_day = min(current + timedelta(days=1), end)
        yield current.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")
        current = next_day


def _parse_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Unsupported time format: {value}")


def _format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _date_from_time(value: str) -> str:
    return _parse_time(value).strftime("%Y-%m-%d")


def _optional_config_date(msnoise_cfg: dict, key: str, fallback: str) -> str:
    raw_value = str(msnoise_cfg.get(key) or "").strip()
    if not raw_value:
        return fallback
    return _date_from_time(raw_value)


def _iter_time_chunks(start_time: str, end_time: str, chunk_hours: int):
    start = _parse_time(start_time)
    end = _parse_time(end_time)
    current = start
    while current < end:
        next_time = min(current + timedelta(hours=chunk_hours), end)
        yield _format_time(current), _format_time(next_time)
        current = next_time


def _fetch_text_url(url: str, timeout: int) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception as urlopen_error:
        result = subprocess.run(
            ["curl", "-L", "-sS", "-f", "--max-time", str(timeout), url],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"URL request failed with urllib and curl: {url}. "
                f"urllib={urlopen_error}; curl_stderr={result.stderr.strip()}"
            ) from urlopen_error
        return result.stdout


def load_station_metadata(msnoise_cfg: dict) -> list[dict[str, object]]:
    manual_rows = msnoise_cfg.get("station_metadata")
    if manual_rows:
        return [
            {
                "net": row["network"],
                "sta": row["station"],
                "lon": float(row["longitude"]),
                "lat": float(row["latitude"]),
                "elev": float(row["elevation"]),
            }
            for row in manual_rows
        ]

    station_service_url = str(require_value(msnoise_cfg, "station_service_url", "dvv_calculation.msnoise"))
    params = {
        "channel": require_value(msnoise_cfg, "channels", "dvv_calculation.msnoise"),
        "starttime": require_value(msnoise_cfg, "start_date", "dvv_calculation.msnoise"),
        "endtime": require_value(msnoise_cfg, "end_date", "dvv_calculation.msnoise"),
        "level": "station",
        "format": "text",
        "nodata": "404",
    }
    network = str(msnoise_cfg.get("network", "")).strip()
    stations = msnoise_cfg.get("stations") or []
    location = str(msnoise_cfg.get("location", "")).strip()
    if network:
        params["network"] = network
    if stations:
        params["station"] = ",".join(str(station) for station in stations)
    else:
        bounds = require_mapping(msnoise_cfg, "geographic_bounds")
        for key in ("minlatitude", "maxlatitude", "minlongitude", "maxlongitude"):
            params[key] = require_value(bounds, key, "dvv_calculation.msnoise.geographic_bounds")
    if location:
        params["location"] = location
    url = station_service_url + "?" + urllib.parse.urlencode(params)
    timeout = int(require_value(msnoise_cfg, "download_timeout", "dvv_calculation.msnoise"))
    print(f"Querying station metadata: {url}", flush=True)
    payload = _fetch_text_url(url, timeout)

    rows_by_station: dict[tuple[str, str], dict[str, object]] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        key = (parts[0], parts[1])
        rows_by_station[key] = {
            "net": parts[0],
            "sta": parts[1],
            "lat": float(parts[2]),
            "lon": float(parts[3]),
            "elev": float(parts[4]),
        }

    rows = list(rows_by_station.values())
    if not rows:
        raise ValueError(f"No station metadata returned by {station_service_url}")
    return rows


def _write_sds_trace(trace, sds_root: Path) -> Path:
    from obspy import UTCDateTime
    from obspy import read

    start_time = trace.stats.starttime
    current_time = UTCDateTime(start_time.date)
    written_path = None

    while current_time < trace.stats.endtime:
        next_day = current_time + 86400
        day_slice = trace.slice(starttime=current_time, endtime=next_day - 0.000001)
        if day_slice.stats.npts > 0:
            year = str(current_time.year)
            net = day_slice.stats.network
            sta = day_slice.stats.station
            loc = day_slice.stats.location or ""
            chan = day_slice.stats.channel
            save_dir = sds_root / year / net / sta
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{net}.{sta}.{loc}.{chan}.D.{year}.{current_time.julday:03d}"
            written_path = save_dir / fname
            if written_path.exists():
                combined = read(str(written_path))
                combined += day_slice
                combined.merge(method=1, fill_value="interpolate")
                tmp_path = written_path.with_name(written_path.name + ".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                combined.write(str(tmp_path), format="MSEED")
                tmp_path.replace(written_path)
            else:
                day_slice.write(str(written_path), format="MSEED")
        current_time = next_day

    if written_path is None:
        raise ValueError(f"Trace has no samples to write: {trace.id}")
    return written_path


def _download_msnoise_sds(msnoise_cfg: dict, project_dir: Path) -> tuple[Path, list[dict[str, object]]]:
    import obspy

    dataselect_url = str(require_value(msnoise_cfg, "dataselect_url", "dvv_calculation.msnoise"))
    location = str(require_value(msnoise_cfg, "location", "dvv_calculation.msnoise"))
    channels = str(require_value(msnoise_cfg, "channels", "dvv_calculation.msnoise"))
    start_date = str(require_value(msnoise_cfg, "start_date", "dvv_calculation.msnoise"))
    end_date = str(require_value(msnoise_cfg, "end_date", "dvv_calculation.msnoise"))
    timeout = int(require_value(msnoise_cfg, "download_timeout", "dvv_calculation.msnoise"))
    curl_retries = int(require_value(msnoise_cfg, "curl_retries", "dvv_calculation.msnoise"))
    chunk_hours = int(require_value(msnoise_cfg, "download_chunk_hours", "dvv_calculation.msnoise"))
    no_data_behavior = str(require_value(msnoise_cfg, "no_data_behavior", "dvv_calculation.msnoise")).lower()
    skip_existing_downloads = bool(msnoise_cfg.get("skip_existing_downloads", False))
    if no_data_behavior not in {"skip", "fail"}:
        raise ValueError('dvv_calculation.msnoise.no_data_behavior must be "skip" or "fail"')
    if chunk_hours <= 0:
        raise ValueError("dvv_calculation.msnoise.download_chunk_hours must be positive")
    sds_root = project_dir / str(require_value(msnoise_cfg, "sds_folder", "dvv_calculation.msnoise"))
    raw_root = project_dir / "raw_mseed"
    sds_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    station_rows = load_station_metadata(msnoise_cfg)
    download_targets = [(str(row["net"]), str(row["sta"])) for row in station_rows]
    downloaded_by_station = {f"{network}.{station}": 0 for network, station in download_targets}

    for network, station in download_targets:
        for chunk_start, chunk_end in _iter_time_chunks(start_date, end_date, chunk_hours):
            raw_path = raw_root / f"{network}.{station}.{chunk_start.replace(':', '').replace('-', '')}.mseed"
            part_path = raw_path.with_suffix(raw_path.suffix + ".part")
            if skip_existing_downloads and raw_path.exists() and raw_path.stat().st_size > 0:
                print(f"Using existing {network}.{station} {chunk_start} to {chunk_end}", flush=True)
                downloaded_by_station[f"{network}.{station}"] += 1
                continue
            query = dataselect_url + "?" + urllib.parse.urlencode(
                {
                    "network": network,
                    "station": station,
                    "location": location,
                    "channel": channels,
                    "starttime": chunk_start,
                    "endtime": chunk_end,
                }
            )
            command = [
                "curl",
                "-L",
                "-sS",
                "--retry",
                str(curl_retries),
                "--max-time",
                str(timeout),
                "-o",
                str(part_path),
                "-w",
                "%{http_code}",
                query,
            ]
            print(f"Downloading {network}.{station} {chunk_start} to {chunk_end}", flush=True)
            if part_path.exists():
                part_path.unlink()
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            http_code = result.stdout.strip()
            if http_code == "204" and result.returncode == 0:
                message = (
                    f"No waveform data returned for {network}.{station} "
                    f"{chunk_start} to {chunk_end} (HTTP 204)."
                )
                if no_data_behavior == "skip":
                    print(f"WARNING: {message} Skipping this chunk.", flush=True)
                    if part_path.exists():
                        part_path.unlink()
                    continue
                raise RuntimeError(message)
            if result.returncode != 0 or http_code != "200" or not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError(
                    "Waveform download failed: "
                    f"station={network}.{station}, start={chunk_start}, end={chunk_end}, "
                    f"http={http_code}, returncode={result.returncode}, stderr={result.stderr.strip()}"
                )

            try:
                stream = obspy.read(str(part_path))
            except Exception as exc:
                raise RuntimeError(
                    "Downloaded waveform could not be read by ObsPy: "
                    f"station={network}.{station}, start={chunk_start}, end={chunk_end}, "
                    f"path={part_path}, size={part_path.stat().st_size}"
                ) from exc
            part_path.replace(raw_path)
            stream.merge(method=1, fill_value="interpolate")
            for trace in stream:
                _write_sds_trace(trace, sds_root)
            downloaded_by_station[f"{network}.{station}"] += 1

    missing_stations = [station for station, count in downloaded_by_station.items() if count == 0]
    if missing_stations:
        raise RuntimeError(
            "No waveform chunks were downloaded for station(s): "
            + ", ".join(missing_stations)
            + ". Check the configured date range, location, and channel."
        )

    return sds_root, station_rows


def _set_msnoise_config(cursor, name: str, value: object) -> None:
    cursor.execute("INSERT OR REPLACE INTO config (name, value) VALUES (?, ?)", (name, str(value)))


def _write_msnoise_processing_config(cursor, msnoise_cfg: dict) -> None:
    startdate = str(msnoise_cfg.get("msnoise_startdate") or _date_from_time(str(require_value(msnoise_cfg, "start_date", "dvv_calculation.msnoise"))))
    enddate = str(msnoise_cfg.get("msnoise_enddate") or _date_from_time(str(require_value(msnoise_cfg, "end_date", "dvv_calculation.msnoise"))))
    startdate = _date_from_time(startdate)
    enddate = _date_from_time(enddate)
    ref_begin = _optional_config_date(msnoise_cfg, "ref_begin", startdate)
    ref_end = _optional_config_date(msnoise_cfg, "ref_end", enddate)
    if ref_end < ref_begin:
        raise ValueError(f"dvv_calculation.msnoise.ref_end ({ref_end}) is earlier than ref_begin ({ref_begin})")
    _set_msnoise_config(cursor, "startdate", startdate)
    _set_msnoise_config(cursor, "enddate", enddate)
    _set_msnoise_config(cursor, "ref_begin", ref_begin)
    _set_msnoise_config(cursor, "ref_end", ref_end)
    _set_msnoise_config(cursor, "autocorr", require_configured(msnoise_cfg, "autocorr", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "components_to_compute", require_value(msnoise_cfg, "components_to_compute", "dvv_calculation.msnoise"))
    _set_msnoise_config(
        cursor,
        "components_to_compute_single_station",
        require_value(msnoise_cfg, "components_to_compute_single_station", "dvv_calculation.msnoise"),
    )
    _set_msnoise_config(cursor, "mov_stack", require_value(msnoise_cfg, "mov_stack", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "analysis_duration", require_value(msnoise_cfg, "analysis_duration", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "corr_duration", require_value(msnoise_cfg, "corr_duration", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "cc_sampling_rate", require_value(msnoise_cfg, "cc_sampling_rate", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "whitening", require_value(msnoise_cfg, "whitening", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_lag", require_value(msnoise_cfg, "dtt_lag", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_v", require_value(msnoise_cfg, "dtt_v", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_width", require_value(msnoise_cfg, "dtt_width", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_sides", require_value(msnoise_cfg, "dtt_sides", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_minlag", require_value(msnoise_cfg, "dtt_minlag", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_mincoh", require_value(msnoise_cfg, "dtt_mincoh", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_maxerr", require_value(msnoise_cfg, "dtt_maxerr", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "dtt_maxdt", require_value(msnoise_cfg, "dtt_maxdt", "dvv_calculation.msnoise"))
    _set_msnoise_config(cursor, "stack_method", require_value(msnoise_cfg, "stack_method", "dvv_calculation.msnoise"))

    extra_config = require_mapping(msnoise_cfg, "extra_msnoise_config")
    for name, value in extra_config.items():
        _set_msnoise_config(cursor, str(name), value)

    cursor.execute("DELETE FROM filters")
    for filter_cfg in require_value(msnoise_cfg, "filters", "dvv_calculation.msnoise"):
        cursor.execute(
            """
            INSERT INTO filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                filter_cfg["ref"],
                filter_cfg["low"],
                filter_cfg["mwcs_low"],
                filter_cfg["high"],
                filter_cfg["mwcs_high"],
                filter_cfg["rms_threshold"],
                filter_cfg["mwcs_wlen"],
                filter_cfg["mwcs_step"],
            ),
        )


def sync_msnoise_project_config(msnoise_cfg: dict, project_dir: Path) -> None:
    db_path = project_dir / "msnoise.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"MSNoise database does not exist: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        _set_msnoise_config(cursor, "data_folder", str(project_dir))
        _set_msnoise_config(cursor, "data_structure", require_value(msnoise_cfg, "data_structure", "dvv_calculation.msnoise"))
        _set_msnoise_config(cursor, "data_type", require_value(msnoise_cfg, "data_type", "dvv_calculation.msnoise"))
        _write_msnoise_processing_config(cursor, msnoise_cfg)
        conn.commit()
    finally:
        conn.close()


def clear_msnoise_jobs(project_dir: Path) -> None:
    db_path = project_dir / "msnoise.sqlite"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM jobs")
        conn.commit()
    finally:
        conn.close()


def _scan_msnoise_project(msnoise_cfg: dict, project_dir: Path, sds_root: Path, station_rows: list[dict[str, object]]) -> None:
    import obspy

    db_path = project_dir / "msnoise.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"MSNoise database was not initialized: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        _set_msnoise_config(cursor, "data_folder", str(project_dir))
        _set_msnoise_config(cursor, "data_structure", require_value(msnoise_cfg, "data_structure", "dvv_calculation.msnoise"))
        _set_msnoise_config(cursor, "data_type", require_value(msnoise_cfg, "data_type", "dvv_calculation.msnoise"))
        _write_msnoise_processing_config(cursor, msnoise_cfg)

        cursor.execute("DELETE FROM stations")
        station_coordinates = str(require_value(msnoise_cfg, "station_coordinates", "dvv_calculation.msnoise"))
        station_instrument = str(require_value(msnoise_cfg, "station_instrument", "dvv_calculation.msnoise"))
        for row in station_rows:
            cursor.execute(
                """
                INSERT INTO stations (net, sta, X, Y, altitude, coordinates, instrument, used)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (row["net"], row["sta"], row["lon"], row["lat"], row["elev"], station_coordinates, station_instrument),
            )

        cursor.execute("DELETE FROM data_availability")
        count = 0
        for file_path in sorted(sds_root.rglob("*")):
            if not file_path.is_file():
                continue
            stream = obspy.read(str(file_path), headonly=True)
            trace = stream[0]
            cursor.execute(
                """
                INSERT INTO data_availability
                (net, sta, comp, path, file, starttime, endtime, data_duration, gaps_duration, samplerate, flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'N')
                """,
                (
                    trace.stats.network,
                    trace.stats.station,
                    trace.stats.channel,
                    str(file_path.parent.relative_to(project_dir)),
                    file_path.name,
                    trace.stats.starttime.datetime,
                    trace.stats.endtime.datetime,
                    trace.stats.endtime - trace.stats.starttime,
                    trace.stats.sampling_rate,
                ),
            )
            count += 1
        conn.commit()
        print(f"MSNoise SDS scan registered {count} files.")
    finally:
        conn.close()


def run_msnoise_project_setup(msnoise_cfg: dict, project_dir: Path) -> None:
    if not require_bool(msnoise_cfg, "prepare_project", "dvv_calculation.msnoise"):
        return

    if require_bool(msnoise_cfg, "reset_project_dir", "dvv_calculation.msnoise") and project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    sds_root, station_rows = _download_msnoise_sds(msnoise_cfg, project_dir)
    db_tech = str(require_value(msnoise_cfg, "msnoise_db_tech", "dvv_calculation.msnoise"))
    if not (project_dir / "msnoise.sqlite").exists():
        subprocess.run(["msnoise", "db", "init", "--tech", db_tech], cwd=project_dir, check=True)
    _scan_msnoise_project(msnoise_cfg, project_dir, sds_root, station_rows)


def validate_msnoise_outputs(cfg: dict, project_dir: Path) -> dict[str, object]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict) or not require_bool(msnoise_cfg, "validate_outputs", "dvv_calculation.msnoise"):
        return {}
    if not require_bool(msnoise_cfg, "run_commands", "dvv_calculation.msnoise"):
        return {}
    if require_bool(msnoise_cfg, "dry_run", "dvv_calculation.msnoise"):
        return {}

    ref_files = sorted((project_dir / "STACKS").glob("*/REF/*/*"))
    mwcs_files = sorted((project_dir / "MWCS").glob("*/*/*/*/*.txt"))
    dtt_files = sorted((project_dir / "DTT").glob("*/*/*/*.txt"))
    validation = {
        "ref_files": len(ref_files),
        "mwcs_files": len(mwcs_files),
        "dtt_files": len(dtt_files),
        "first_dtt_file": dtt_files[0] if dtt_files else None,
    }

    missing: list[str] = []
    if require_bool(msnoise_cfg, "require_ref", "dvv_calculation.msnoise") and not ref_files:
        missing.append("REF stacks")
    if require_bool(msnoise_cfg, "require_mwcs", "dvv_calculation.msnoise") and not mwcs_files:
        missing.append("MWCS files")
    if require_bool(msnoise_cfg, "require_dtt", "dvv_calculation.msnoise") and not dtt_files:
        missing.append("DTT files")
    if missing:
        raise RuntimeError(
            "MSNoise output validation failed. Missing: "
            + ", ".join(missing)
            + f". Project directory: {project_dir}"
        )

    db_path = project_dir / "msnoise.sqlite"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "select jobtype, flag, count(*) from jobs group by jobtype, flag order by jobtype, flag"
            )
            validation["job_summary"] = ["|".join(str(item) for item in row) for row in cursor.fetchall()]
        finally:
            conn.close()

    return validation


def validate_mcmc_outputs(cfg: dict, outputs: dict[str, Path]) -> dict[str, object]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    mcmc_cfg = dvv_cfg.get("mcmc")
    if not isinstance(mcmc_cfg, dict) or not require_bool(mcmc_cfg, "validate_outputs", "dvv_calculation.mcmc"):
        return {}

    summary_path = outputs.get("summary")
    likelihood_path = outputs.get("likelihood")
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(f"MCMC summary output is missing: {summary_path}")
    if likelihood_path is None or not likelihood_path.exists():
        raise FileNotFoundError(f"MCMC likelihood output is missing: {likelihood_path}")

    rows: list[dict[str, str]]
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    min_rows = int(require_value(mcmc_cfg, "min_summary_rows", "dvv_calculation.mcmc"))
    if len(rows) < min_rows:
        raise RuntimeError(f"MCMC summary has {len(rows)} rows; expected at least {min_rows}: {summary_path}")

    numeric_columns = ["mean", "median", "p025", "p975"]
    bad_rows: list[str] = []
    for row in rows:
        for column in numeric_columns:
            value = _as_float(row.get(column))
            if value is None:
                bad_rows.append(row.get("date", "<unknown-date>"))
                break

    if bad_rows:
        raise RuntimeError(
            "MCMC summary contains non-finite values for dates: "
            + ", ".join(bad_rows[:10])
            + f". Summary path: {summary_path}"
        )

    return {
        "mcmc_summary_rows": len(rows),
        "mcmc_first_date": rows[0].get("date") if rows else None,
        "mcmc_last_date": rows[-1].get("date") if rows else None,
        "mcmc_summary": summary_path,
        "mcmc_likelihood": likelihood_path,
    }


def _read_table_rows(path: Path) -> list[dict[str, str]]:
    input_files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    rows: list[dict[str, str]] = []

    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
            reader = csv.DictReader(handle, dialect=dialect)
            for row in reader:
                row["_source_file"] = str(file_path)
                rows.append(row)

    return rows


def build_observation_system(cfg: dict) -> ObservationSystem:
    paths = require_mapping(cfg, "paths")
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    mcmc_cfg = dvv_cfg.get("mcmc")
    if not isinstance(mcmc_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.mcmc")

    input_path = resolve_path(cfg, require_value(paths, "dvv_observation_input", "paths"))

    date_column = str(require_value(mcmc_cfg, "date_column", "dvv_calculation.mcmc"))
    observation_column = str(require_value(mcmc_cfg, "observation_column", "dvv_calculation.mcmc"))
    variance_column = str(require_configured(mcmc_cfg, "variance_column", "dvv_calculation.mcmc"))
    error_column = str(require_configured(mcmc_cfg, "error_column", "dvv_calculation.mcmc"))
    pair_column = str(require_configured(mcmc_cfg, "pair_column", "dvv_calculation.mcmc"))
    pair_selector = str(require_configured(mcmc_cfg, "pair_selector", "dvv_calculation.mcmc"))
    weight_floor = float(require_value(mcmc_cfg, "weight_floor", "dvv_calculation.mcmc"))

    rows = _read_table_rows(input_path)
    if not rows:
        raise ValueError(f"No observations found in {input_path}")

    date_values: list[str] = []
    obs_values: list[float] = []
    raw_weights: list[float] = []

    for row in rows:
        if pair_selector:
            row_pair = _first_present(row, [pair_column, "Pairs", "pair", "station_pair"])
            if row_pair != pair_selector:
                continue

        date_value = _first_present(row, [date_column, "Date", "date", "day", "startdate", "enddate", "time"])
        obs_value = _as_float(_first_present(row, [observation_column, "M", "dvv", "dt_over_t", "dtt", "dt"]))
        variance_value = _as_float(_first_present(row, [variance_column, "variance", "var"]))
        error_value = _as_float(_first_present(row, [error_column, "EM", "error", "err", "sigma", "std"]))
        corr_value = _as_float(_first_present(row, ["corr", "coh", "coherence", "cc"]))

        if date_value is None or obs_value is None:
            continue

        if variance_value is not None and variance_value > 0:
            weight = 1.0 / max(variance_value, weight_floor)
        elif error_value is not None and error_value > 0:
            weight = 1.0 / max(error_value * error_value, weight_floor)
        elif corr_value is not None and corr_value > 0:
            weight = max(corr_value, weight_floor)
        else:
            weight = 1.0

        date_values.append(date_value[:10])
        obs_values.append(obs_value)
        raw_weights.append(weight)

    if not obs_values:
        raise ValueError(f"No usable observations found in {input_path}")

    dates = sorted(set(date_values))
    date_index = {date: idx for idx, date in enumerate(dates)}
    g = np.zeros((len(obs_values), len(dates)), dtype=float)
    for row_idx, date_value in enumerate(date_values):
        g[row_idx, date_index[date_value]] = 1.0

    return ObservationSystem(
        dates=dates,
        observations=np.asarray(obs_values, dtype=float),
        g=g,
        weights=np.asarray(raw_weights, dtype=float),
    )


def import_mcmc_class(cfg: dict):
    paths = require_mapping(cfg, "paths")
    repo_path = resolve_path(cfg, require_value(paths, "mcmc_dvv_repo", "paths"))
    sys.path.insert(0, str(repo_path))
    try:
        from mcmc_inversion import MarkovChainMonteCarlo
    finally:
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass
    return MarkovChainMonteCarlo


def run_mcmc_backend(cfg: dict) -> dict[str, Path]:
    paths = require_mapping(cfg, "paths")
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    mcmc_cfg = dvv_cfg.get("mcmc")
    if not isinstance(mcmc_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.mcmc")

    random_seed = require_value(mcmc_cfg, "random_seed", "dvv_calculation.mcmc")
    if random_seed is not None:
        np.random.seed(int(random_seed))

    system = build_observation_system(cfg)
    iterations = int(require_value(mcmc_cfg, "iterations", "dvv_calculation.mcmc"))
    if iterations <= 0 or iterations % 100 != 0:
        raise ValueError("dvv_calculation.mcmc.iterations must be positive and divisible by 100")

    burn_in = int(require_value(mcmc_cfg, "burn_in", "dvv_calculation.mcmc"))
    if burn_in < 0 or burn_in >= iterations:
        raise ValueError("dvv_calculation.mcmc.burn_in must be >= 0 and smaller than iterations")

    mcmc_class = import_mcmc_class(cfg)
    inversion = mcmc_class(
        system.observations,
        system.g,
        system.weights,
        len(system.dates),
        float(require_value(mcmc_cfg, "prior_low", "dvv_calculation.mcmc")),
        float(require_value(mcmc_cfg, "prior_high", "dvv_calculation.mcmc")),
    )
    proposal_std = float(require_value(mcmc_cfg, "proposal_std", "dvv_calculation.mcmc"))
    distribution, likelihood = inversion.do_mcmc(iterations, proposal_std)
    posterior = distribution[:, burn_in:]

    output_dir = (
        resolve_path(cfg, require_value(paths, "output_dir", "paths"))
        / str(require_value(paths, "output_subdir", "paths"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dvv_mcmc_summary.csv"
    likelihood_path = output_dir / "likelihood.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "mean", "median", "p025", "p975", "n_observations"])
        counts = np.sum(system.g, axis=0).astype(int)
        for idx, date_value in enumerate(system.dates):
            samples = posterior[idx, :]
            writer.writerow(
                [
                    date_value,
                    f"{np.mean(samples):.10g}",
                    f"{np.median(samples):.10g}",
                    f"{np.percentile(samples, 2.5):.10g}",
                    f"{np.percentile(samples, 97.5):.10g}",
                    int(counts[idx]),
                ]
            )

    with likelihood_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "likelihood"])
        for idx, value in enumerate(likelihood):
            writer.writerow([idx, f"{value:.10g}"])

    return {"summary": summary_path, "likelihood": likelihood_path}


def run_msnoise_backend(cfg: dict) -> list[list[str]]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.msnoise")

    commands = require_value(msnoise_cfg, "commands", "dvv_calculation.msnoise")
    normalized_commands = [[str(part) for part in command] for command in commands]

    if not require_bool(msnoise_cfg, "run_commands", "dvv_calculation.msnoise"):
        return normalized_commands

    project_dir = resolve_path(cfg, require_value(msnoise_cfg, "project_dir", "dvv_calculation.msnoise"))
    if require_bool(msnoise_cfg, "dry_run", "dvv_calculation.msnoise"):
        return normalized_commands

    run_msnoise_project_setup(msnoise_cfg, project_dir)
    sync_msnoise_project_config(msnoise_cfg, project_dir)
    if require_bool(msnoise_cfg, "clear_jobs_on_run", "dvv_calculation.msnoise"):
        clear_msnoise_jobs(project_dir)

    for command in normalized_commands:
        subprocess.run(command, cwd=project_dir, check=True)

    return normalized_commands


def run_dvv_calculation(cfg: dict, method_override: str | None = None) -> DvvRunResult:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.msnoise")
    method = method_override or str(require_value(msnoise_cfg, "method", "dvv_calculation.msnoise"))
    allowed = {"msnoise", "mcmc", "both"}
    if method not in allowed:
        raise ValueError(f"Unsupported dv/v method: {method}")

    msnoise_commands: list[list[str]] = []
    mcmc_outputs: dict[str, Path] = {}
    validation: dict[str, object] = {}

    if method in {"msnoise", "both"}:
        msnoise_commands = run_msnoise_backend(cfg)
        msnoise_cfg = require_mapping(dvv_cfg, "msnoise")
        project_dir = resolve_path(cfg, require_value(msnoise_cfg, "project_dir", "dvv_calculation.msnoise"))
        validation.update(validate_msnoise_outputs(cfg, project_dir))

    if method in {"mcmc", "both"}:
        mcmc_outputs = run_mcmc_backend(cfg)
        validation.update(validate_mcmc_outputs(cfg, mcmc_outputs))

    return DvvRunResult(
        method=method,
        msnoise_commands=msnoise_commands,
        mcmc_outputs=mcmc_outputs,
        validation=validation,
    )


def print_result(result: DvvRunResult) -> None:
    print(f"dv/v method: {result.method}")
    if result.msnoise_commands:
        print("MSNoise commands configured:")
        for command in result.msnoise_commands:
            print("  " + " ".join(command))
    if result.mcmc_outputs:
        print(f"MCMC summary: {result.mcmc_outputs['summary']}")
        print(f"MCMC likelihood: {result.mcmc_outputs['likelihood']}")
    if result.validation:
        print("Validation:")
        if result.validation.get("ref_files") is not None:
            print(f"  REF files: {result.validation['ref_files']}")
        if result.validation.get("mwcs_files") is not None:
            print(f"  MWCS files: {result.validation['mwcs_files']}")
        if result.validation.get("dtt_files") is not None:
            print(f"  DTT files: {result.validation['dtt_files']}")
        if result.validation.get("first_dtt_file"):
            print(f"  First DTT file: {result.validation['first_dtt_file']}")
        if result.validation.get("job_summary"):
            print("  Job summary:")
            for row in result.validation["job_summary"]:
                print(f"    {row}")
        if result.validation.get("mcmc_summary_rows") is not None:
            print(f"  MCMC summary rows: {result.validation['mcmc_summary_rows']}")
            print(f"  MCMC date range: {result.validation['mcmc_first_date']} to {result.validation['mcmc_last_date']}")
