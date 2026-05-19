from __future__ import annotations

import shutil
import sqlite3
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MSNOISE_PROJECT_DIR = "outputs/dvv_calculation/testing"
DATASELECT_URL = "https://service.iris.edu/fdsnws/dataselect/1/query"
STATION_SERVICE_URL = "https://service.iris.edu/fdsnws/station/1/query"
SDS_FOLDER = "SDS"
DATA_STRUCTURE = "SDS"
MSNOISE_DB_TECH = "1"
STATION_COORDINATES = "DEG"
STATION_INSTRUMENT = "INST"
LOCATION = "*"


@dataclass(frozen=True)
class DvvRunResult:
    msnoise_commands: list[list[str]]
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
    if network:
        params["network"] = network
    if stations:
        params["station"] = ",".join(str(station) for station in stations)
    else:
        bounds = require_mapping(msnoise_cfg, "geographic_bounds")
        for key in ("minlatitude", "maxlatitude", "minlongitude", "maxlongitude"):
            params[key] = require_value(bounds, key, "dvv_calculation.msnoise.geographic_bounds")
    params["location"] = LOCATION
    url = STATION_SERVICE_URL + "?" + urllib.parse.urlencode(params)
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
        raise ValueError(f"No station metadata returned by {STATION_SERVICE_URL}")
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
    sds_root = project_dir / SDS_FOLDER
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
            query = DATASELECT_URL + "?" + urllib.parse.urlencode(
                {
                    "network": network,
                    "station": station,
                    "location": LOCATION,
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
        _set_msnoise_config(cursor, "data_structure", DATA_STRUCTURE)
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
        _set_msnoise_config(cursor, "data_structure", DATA_STRUCTURE)
        _set_msnoise_config(cursor, "data_type", require_value(msnoise_cfg, "data_type", "dvv_calculation.msnoise"))
        _write_msnoise_processing_config(cursor, msnoise_cfg)

        cursor.execute("DELETE FROM stations")
        for row in station_rows:
            cursor.execute(
                """
                INSERT INTO stations (net, sta, X, Y, altitude, coordinates, instrument, used)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (row["net"], row["sta"], row["lon"], row["lat"], row["elev"], STATION_COORDINATES, STATION_INSTRUMENT),
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
    if not (project_dir / "msnoise.sqlite").exists():
        subprocess.run(["msnoise", "db", "init", "--tech", MSNOISE_DB_TECH], cwd=project_dir, check=True)
    _scan_msnoise_project(msnoise_cfg, project_dir, sds_root, station_rows)


def validate_msnoise_outputs(cfg: dict, project_dir: Path) -> dict[str, object]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict) or not require_bool(msnoise_cfg, "validate_outputs", "dvv_calculation.msnoise"):
        return {}
    if not require_bool(msnoise_cfg, "run_commands", "dvv_calculation.msnoise"):
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


def run_msnoise_backend(cfg: dict) -> list[list[str]]:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.msnoise")

    commands = require_value(msnoise_cfg, "commands", "dvv_calculation.msnoise")
    normalized_commands = [[str(part) for part in command] for command in commands]

    if not require_bool(msnoise_cfg, "run_commands", "dvv_calculation.msnoise"):
        return normalized_commands

    project_dir = resolve_path(cfg, MSNOISE_PROJECT_DIR)

    run_msnoise_project_setup(msnoise_cfg, project_dir)
    sync_msnoise_project_config(msnoise_cfg, project_dir)
    if require_bool(msnoise_cfg, "clear_jobs_on_run", "dvv_calculation.msnoise"):
        clear_msnoise_jobs(project_dir)

    for command in normalized_commands:
        subprocess.run(command, cwd=project_dir, check=True)

    return normalized_commands


def run_dvv_calculation(cfg: dict) -> DvvRunResult:
    dvv_cfg = require_mapping(cfg, "dvv_calculation")
    msnoise_cfg = dvv_cfg.get("msnoise")
    if not isinstance(msnoise_cfg, dict):
        raise KeyError("Missing required config object: dvv_calculation.msnoise")

    msnoise_commands = run_msnoise_backend(cfg)
    project_dir = resolve_path(cfg, MSNOISE_PROJECT_DIR)
    validation = validate_msnoise_outputs(cfg, project_dir)

    return DvvRunResult(msnoise_commands=msnoise_commands, validation=validation)


def print_result(result: DvvRunResult) -> None:
    print("dv/v method: msnoise")
    if result.msnoise_commands:
        print("MSNoise commands configured:")
        for command in result.msnoise_commands:
            print("  " + " ".join(command))
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
