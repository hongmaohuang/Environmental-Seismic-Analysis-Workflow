import numpy as np
import pandas as pd
import warnings
from scipy.signal import fftconvolve
from scipy.special import erfc, erf

##############################################
### Data Reading and Time Grid Preparation ###
##############################################

def read_groundwater(filepath):
    """
    Read the groundwater input with columns:
    time, temperature, groundwater level [m a.s.l.]
    """
    df = pd.read_csv(filepath, usecols=["time", "temperature", "groundwater level [m a.s.l.]"])
    df["time"] = pd.to_datetime(df["time"])
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["groundwater level [m a.s.l.]"] = pd.to_numeric(df["groundwater level [m a.s.l.]"], errors="coerce")
    return (
        df.rename(
            columns={
                "temperature": "well_temp_c",
                "groundwater level [m a.s.l.]": "gwl_m_asl",
            }
        )
        .dropna(subset=["gwl_m_asl", "well_temp_c"])
        .set_index("time")
        .sort_index()
    )

def read_atmospheric_pressure(filepath):
    """
    Read the atmospheric pressure input with columns:
    time, atmosphere pressure (Pa)
    """
    df = pd.read_csv(filepath, usecols=["time", "atmosphere pressure"])
    df["time"] = pd.to_datetime(df["time"])
    df["patm_pa"] = pd.to_numeric(df["atmosphere pressure"], errors="coerce")
    df = df.set_index("time").sort_index()
    return df[["patm_pa"]].dropna()

def read_snow(filepath):
    """
    Read snow input with columns:
    time, snow depth, and optionally snow cover.
    The snow depth is expected to be in meters, and snow cover is a percentage (0-100)
    thinking of getting rid of the snow cover column.
    """
    df = pd.read_csv(filepath)
    required_columns = ["time", "snow depth"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing the snow columns in {filepath}: {missing_columns}")

    df["time"] = pd.to_datetime(df["time"])
    df["snow depth"] = pd.to_numeric(df["snow depth"], errors="coerce")
    if "snow cover" in df.columns:
        df["snow cover"] = pd.to_numeric(df["snow cover"], errors="coerce")

    if df["snow depth"].notna().sum() == 0:
        raise ValueError(
            f"Standardized snow-depth column contains no usable values: {filepath}"
        )

    out = (
        df.rename(
            columns={
                "time": "datetime",
                "snow depth": "snow_depth_m",
                "snow cover": "snow_cover",
            }
        )
        .set_index("datetime")
        .sort_index()
    )
    columns = ["snow_depth_m"]
    if "snow_cover" in out.columns:
        columns.append("snow_cover")
    return out[columns]

def infer_median_dt_seconds(index):
    """
    Calculate time step of original inputs
    """
    if len(index) < 2:
        return None
    dt_s = float(np.median(np.diff(index).astype("timedelta64[s]").astype(float)))
    if dt_s <= 0:
        return None
    return dt_s

def warn_if_source_resolution_is_coarser(
    dataset_name,
    filepath,
    source_dt_s,
    target_dt_s,
    target_sampling_label,
    action_message,
):
    """
    Warning users the original temporal resolution is lower than users' setting
    For example, if the original data is daily but users want to have a final result with 1h resolution, then this function will warn users about the mismatch.
    It still runs it but the results should be double-checked or just change the final temporal resolution (resample_rule) in config.toml
    """
    if source_dt_s is None or target_dt_s >= source_dt_s:
        return
    file_line = f"File: {filepath}\n" if filepath is not None else ""
    warnings.warn(
        "Input data resolution mismatch.\n"
        f"Dataset: {dataset_name}\n"
        f"Requested final sampling: {target_sampling_label}\n"
        f"{file_line}"
        f"Action: {action_message}",
        stacklevel=1,
    )

def prepare_time_series(
    source_df,
    start=None,
    end=None,
    rule=None,
    target_index=None,
    interp_method="time",
    dataset_name="input data",
    filepath=None,
):
    """
    align the original time steps to the final one
    if the original one has higher sampling rate, then it will downsample
    if the original one has lower sampling rate, then it will interpolate
    """

    series = source_df.iloc[:, 0].copy()
    series = series.loc[start:end].copy()
    if series.empty:
        raise ValueError(f"No {dataset_name} data in the requested time window")
    if rule is None:
        raise ValueError("resample_rule must be provided")

    if target_index is None:
        series = series.resample(rule).mean().dropna()
        if series.empty:
            raise ValueError(f"{dataset_name} data became empty after resampling")
        return series

    source_dt_s = infer_median_dt_seconds(series.index)
    target_dt_s = infer_median_dt_seconds(target_index)
    if source_dt_s is not None and target_dt_s is not None and source_dt_s > target_dt_s:
        warn_if_source_resolution_is_coarser(
            dataset_name=dataset_name,
            filepath=filepath,
            source_dt_s=source_dt_s,
            target_dt_s=target_dt_s,
            target_sampling_label=rule,
            action_message="the dataset will be interpolated to the final time grid.",
        )
        expanded = series.reindex(series.index.union(target_index)).sort_index()
        expanded = expanded.interpolate(method=interp_method).ffill().bfill()
        return expanded.reindex(target_index).ffill().bfill()
    series = series.resample(rule).mean().dropna()
    if series.empty:
        raise ValueError(f"{dataset_name} data became empty after resampling")
    return series.reindex(target_index).ffill().bfill()

def infer_regular_dt_seconds(index):
    """
    Calculate time step of final results
    for example:
    
    1h >> 3600s
    1D >> 84600s
    """
    t_s = (index - index[0]).total_seconds().astype(float)
    dt_s = float(np.median(np.diff(t_s)))
    if dt_s <= 0:
        raise ValueError("Time index must be increasing")
    return t_s, dt_s


####################################
### Loading response calculation ###
####################################

# def compute_atm_loading_response():
# NTAL
# Non tidal atm loading
# correction

# def compute_tidal_loading_response():
# NTOL
# Non tidal ocean loading
# TLA, TOL

def compute_poroelastic_alpha(skempton_b, undrained_poisson_ratio):
    """
    Calculating elastic constant (alpha)
    You can find the formulation in Roeloffs (1988), equation 19
    """
    denominator = 3.0 * (1.0 - undrained_poisson_ratio)
    if denominator == 0:
        raise ValueError("Invalid undrained Poisson ratio: denominator became zero! Please try different undrained_poisson_ratio.")
    poroelastic_alpha = (skempton_b * (1.0 + undrained_poisson_ratio) / denominator)
    if not 0.0 <= poroelastic_alpha <= 1.0:
        raise ValueError(
            "Computed poroelastic alpha is outside [0, 1]. "
            "Check skempton_b and undrained_poisson_ratio."
        )
    return poroelastic_alpha

def build_default_loadings(
    gwl_m_asl,
    patm_pa,
    rho_w,
    g,
    poroelastic_alpha,
    snow_depth_m=None,
    snow_density_kg_m3=None,
):
    """
    Calculate pressure (Pa) from the gwl and snow depth and density (m) 
    And also grab pressure from air pressure data
    """
    p_gwl_pa = rho_w * g * np.asarray(gwl_m_asl, dtype=float)
    p_atm_pa = np.asarray(patm_pa, dtype=float)
    loadings = [
        {"name": "gwl", "values_pa": p_gwl_pa, "poroelastic_alpha": poroelastic_alpha},
        {"name": "atm", "values_pa": p_atm_pa, "poroelastic_alpha": poroelastic_alpha},
    ]
    if snow_depth_m is not None:
        if snow_density_kg_m3 is None:
            raise ValueError("snow_density_kg_m3 must be provided for snow loading")
        p_snow_pa = snow_density_kg_m3 * g * np.asarray(snow_depth_m, dtype=float)
        loadings.append(
            {
                "name": "snow",
                "values_pa": p_snow_pa,
                "poroelastic_alpha": poroelastic_alpha,
            }
        )
    return loadings

def compute_loading_response(
    loading_pa, depth_m, dt_s, hydraulic_diffusivity_m2_s, poroelastic_alpha
):
    """
    Compute pore-pressure response using the Talwani (2007) equation 4  
    Also the origianl reference- Roeloffs (1988)
    """
    loading_pa = np.asarray(loading_pa, dtype=float)
    if not 0.0 <= poroelastic_alpha <= 1.0:
        raise ValueError("poroelastic_alpha must be between 0 and 1")

    dp = np.zeros_like(loading_pa)
    dp[0] = loading_pa[0]
    dp[1:] = np.diff(loading_pa)

    lag_s = np.arange(len(loading_pa), dtype=float) * dt_s
    kernel = np.zeros(len(loading_pa), dtype=float)
    positive_lag = lag_s > 0
    x = np.zeros(len(loading_pa), dtype=float)
    x[positive_lag] = depth_m / np.sqrt(
        4.0 * hydraulic_diffusivity_m2_s * lag_s[positive_lag]
    )
    kernel[positive_lag] = (
        poroelastic_alpha * erf(x[positive_lag]) + erfc(x[positive_lag])
    )
    if depth_m == 0:
        kernel[0] = 1.0
    else:
        kernel[0] = poroelastic_alpha

    return fftconvolve(dp, kernel, mode="full")[: len(loading_pa)]


##########################################
### Thermoelastic response calculation ###
###########################################

def compute_shear_modulus(youngs_modulus_pa, poisson_ratio):
    """
    Calculating shear modulus from poisson ratio
    """
    denominator = 2.0 * (1.0 + poisson_ratio)
    if denominator == 0:
        raise ValueError("Invalid Poisson ratio: denominator became zero")
    return youngs_modulus_pa / denominator

def compute_second_murnaghan_constant(shear_modulus_pa, m_over_mu_ratio):
    """
    Calculating murnaghan constant. This is for thermoelastic response calculation. 
    The larger the m_over_mu_ratio, the more significant the thermoelastic effects will be on seismic velocity changes. 
    It's like a gain factor that amplifies the influence of temperature-induced strain on seismic velocities.
    """
    return shear_modulus_pa * m_over_mu_ratio

def fit_annual_harmonic(index, values, period_s=365.25 * 86400.0):
    """
    Since the thermoelastic response is mainly driven by the annual temperature cycle, 
    we should fit an annual harmonic to the temperature time series to extract the amplitude and phase of the annual cycle. 
    This program also separates the annual component from any residual temperature variations, which will be in your output CSV.
    """
    values = np.asarray(values, dtype=float)
    index = pd.to_datetime(index)
    t_s = (index - index[0]).total_seconds().astype(float)
    omega = 2.0 * np.pi / period_s
    X = np.column_stack(
        [
            np.ones(len(values), dtype=float),
            np.cos(omega * t_s),
            np.sin(omega * t_s),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    mean_value, a_cos, b_sin = coeffs
    amplitude = float(np.hypot(a_cos, b_sin))
    phase_rad = float(np.arctan2(b_sin, a_cos))
    fitted = X @ coeffs
    return {
        "mean": float(mean_value),
        "a_cos": float(a_cos),
        "b_sin": float(b_sin),
        "amplitude": amplitude,
        "phase_rad": phase_rad,
        "omega": float(omega),
        "fitted": fitted,
    }

def compute_tsai_vsv_thermoelastic_response(
    index,
    temperature_c,
    poisson_ratio,
    shear_modulus_pa,
    second_murnaghan_pa,
    wavenumber_m_inv,
    depth_m,
    thermoelastic_spatial_factor,
    thermal_expansion_coeff_c_inv,
    thermal_diffusivity_m2_s,
    incompetent_layer_thickness_m,
    period_s=365.25 * 86400.0,
):
    """
    Estimate thermoelastic dv/v as a VSV-sensitive Rayleigh-coda response.

    This implements Tsai (2011) equation 3 for A(t), then equation 17 for
    Delta VSV / VSV. The config value thermoelastic_spatial_factor represents
    the sin(kx) term in equation 17; use 1.0 for Tsai's maximum-signal position.
    """
    harmonic = fit_annual_harmonic(index, temperature_c, period_s=period_s)
    t_s = (pd.to_datetime(index) - pd.to_datetime(index)[0]).total_seconds().astype(float)
    omega = harmonic["omega"]
    temp_phase_rad = harmonic["phase_rad"]
    thermal_decay = np.exp(
        -np.sqrt(omega / (2.0 * thermal_diffusivity_m2_s)) * incompetent_layer_thickness_m
    )
    strain_amplitude = (
        ((1.0 + poisson_ratio) / (1.0 - poisson_ratio))
        * wavenumber_m_inv
        * thermal_expansion_coeff_c_inv
        * harmonic["amplitude"]
        * np.sqrt(thermal_diffusivity_m2_s / omega)
        * thermal_decay
    )
    thermoelastic_a_t = strain_amplitude * np.cos(
        omega * t_s
        + temp_phase_rad
        - np.sqrt(omega / (2.0 * thermal_diffusivity_m2_s)) * incompetent_layer_thickness_m
        - np.pi / 4.0
    )
    dvv_vsv_thermoelastic_annual = (
        (second_murnaghan_pa / shear_modulus_pa)
        * thermoelastic_a_t
        * np.exp(-wavenumber_m_inv * depth_m)
        * thermoelastic_spatial_factor
        * (1.0 - 2.0 * poisson_ratio)
    )
    return {
        "temperature_annual_c": pd.Series(harmonic["fitted"], index=index),
        "dvv_vsv_thermoelastic_annual": pd.Series(
            dvv_vsv_thermoelastic_annual,
            index=index,
            name="dvv_vsv_thermoelastic_annual",
        ),
    }


#####################
### Main Workflow ###
#####################

def run_pore_pressure_workflow(
    gwl_csv_path,
    atmospheric_pressure_csv_path,
    output_csv_path,
    snow_csv_path,
    include_snow_loading,
    start,
    end,
    resample_rule,
    depths_m,
    rho_w,
    g,
    skempton_b,
    undrained_poisson_ratio,
    hydraulic_diffusivity_m2_s,
    poisson_ratio,
    youngs_modulus_pa,
    m_over_mu_ratio,
    horizontal_wavenumber_m_inv,
    thermoelastic_spatial_factor,
    thermal_expansion_coeff_c_inv,
    thermal_diffusivity_m2_s,
    incompetent_layer_thickness_m,
    snow_density_kg_m3,
):
    """
    Run the functions above all together to get the result
    Output is a csv file containing modeled pore pressure at different depth
    """
    output_csv_path = str(output_csv_path)

    model_index = pd.date_range(start=start, end=end, freq=resample_rule)
    if len(model_index) < 2:
        raise ValueError("The requested model time grid must contain at least two samples")

    gwl = read_groundwater(gwl_csv_path)
    gwl_rs = prepare_time_series(
        gwl[["gwl_m_asl"]],
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="groundwater",
        filepath=gwl_csv_path,
    )
    _, dt_s = infer_regular_dt_seconds(model_index)
    temp_rs = prepare_time_series(
        gwl[["well_temp_c"]],
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="well temperature",
        filepath=gwl_csv_path,
    )

    # the gwl data aligns with the final temporal resolution (defined by users from RULE)

    patm_monthly = read_atmospheric_pressure(atmospheric_pressure_csv_path)
    patm_rs = prepare_time_series(
        patm_monthly,
        start=start,
        end=end,
        rule=resample_rule,
        target_index=model_index,
        dataset_name="atmospheric pressure",
        filepath=atmospheric_pressure_csv_path,
    )
    
    # bc the gwl data has aligned with the final temporal resolution,
    # the atm can just align with it

    snow_rs = None
    snow_cover_rs = None
    if include_snow_loading:
        if snow_csv_path in (None, ""):
            raise ValueError("snow_csv_path must be provided when include_snow_loading is true")
        snow = read_snow(snow_csv_path)
        snow_rs = prepare_time_series(
            snow[["snow_depth_m"]],
            start=start,
            end=end,
            rule=resample_rule,
            target_index=model_index,
            dataset_name="snow depth",
            filepath=snow_csv_path,
        )
        if "snow_cover" in snow.columns:
            snow_cover_rs = prepare_time_series(
                snow[["snow_cover"]],
                start=start,
                end=end,
                rule=resample_rule,
                target_index=model_index,
                dataset_name="snow cover",
                filepath=snow_csv_path,
            )

    poroelastic_alpha = compute_poroelastic_alpha(
        skempton_b=skempton_b,
        undrained_poisson_ratio=undrained_poisson_ratio,
    )
    shear_modulus_pa = compute_shear_modulus(
        youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    second_murnaghan_pa = compute_second_murnaghan_constant(
        shear_modulus_pa=shear_modulus_pa,
        m_over_mu_ratio=m_over_mu_ratio,
    )
    loadings = build_default_loadings(
        gwl_rs.values,
        patm_rs.values,
        rho_w,
        g,
        poroelastic_alpha,
        snow_depth_m=snow_rs.values if snow_rs is not None else None,
        snow_density_kg_m3=snow_density_kg_m3,
    )

    depths_m = tuple(float(depth) for depth in depths_m)
    if not depths_m:
        raise ValueError("depths_m must contain at least one depth")

    out = pd.DataFrame(index=model_index)
    out["gwl_m_asl"] = gwl_rs.values
    out["well_temp_c"] = temp_rs.values
    out["patm_pa"] = patm_rs.values
    if snow_rs is not None:
        out["snow_depth_m"] = snow_rs.values
        out["snow_density_kg_m3"] = snow_density_kg_m3
    if snow_cover_rs is not None:
        out["snow_cover"] = snow_cover_rs.values
    out["poroelastic_alpha"] = poroelastic_alpha
    out["shear_modulus_pa"] = shear_modulus_pa
    out["second_murnaghan_pa"] = second_murnaghan_pa
    thermoelastic_by_depth = {}
    for depth_m in depths_m:
        thermoelastic_by_depth[depth_m] = compute_tsai_vsv_thermoelastic_response(
            index=model_index,
            temperature_c=temp_rs.values,
            poisson_ratio=poisson_ratio,
            shear_modulus_pa=shear_modulus_pa,
            second_murnaghan_pa=second_murnaghan_pa,
            wavenumber_m_inv=horizontal_wavenumber_m_inv,
            depth_m=depth_m,
            thermoelastic_spatial_factor=thermoelastic_spatial_factor,
            thermal_expansion_coeff_c_inv=thermal_expansion_coeff_c_inv,
            thermal_diffusivity_m2_s=thermal_diffusivity_m2_s,
            incompetent_layer_thickness_m=incompetent_layer_thickness_m,
        )
    first_depth_m = depths_m[0]
    thermoelastic = thermoelastic_by_depth[first_depth_m]
    out["well_temp_annual_c"] = thermoelastic["temperature_annual_c"].values
    out["well_temp_residual_c"] = out["well_temp_c"] - out["well_temp_annual_c"]
    for depth_m, thermoelastic in thermoelastic_by_depth.items():
        out[f"dvv_vsv_thermoelastic_annual_z{int(depth_m)}m"] = (
            thermoelastic["dvv_vsv_thermoelastic_annual"].values
        )

    for loading in loadings:
        out[f"{loading['name']}_loading_pa"] = loading["values_pa"]
        for depth_m in depths_m:
            response = compute_loading_response(
                loading_pa=loading["values_pa"],
                depth_m=depth_m,
                dt_s=dt_s,
                hydraulic_diffusivity_m2_s=hydraulic_diffusivity_m2_s,
                poroelastic_alpha=loading["poroelastic_alpha"],
            )
            out[f"Pp_{loading['name']}_z{int(depth_m)}m_pa"] = response

    for depth_m in depths_m:
        total_pp = np.zeros(len(out), dtype=float)
        for loading in loadings:
            total_pp += out[f"Pp_{loading['name']}_z{int(depth_m)}m_pa"].values
        out[f"Pp_total_z{int(depth_m)}m_pa"] = total_pp
        out[f"dPp_total_z{int(depth_m)}m_pa"] = total_pp - np.nanmean(total_pp)

    out.to_csv(output_csv_path, index=True, index_label="datetime")
    return out
