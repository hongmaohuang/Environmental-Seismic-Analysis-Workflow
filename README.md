# Environmental-Seismic-Analysis-Workflow
Hong-Mao Huang, 2026
</p>
Department of Earth Science, University of Colorado Boulder, CO, USA


## Introduction
This workflow supports environmental seismic analysis, including dv/v computation, environmental pressure modeling, pressure-to-dv/v transfer functions, and correlation analysis.

The dv/v computation is built on the Python package MSNoise (Lecocq et al., 2014). The pressure-modeling components follow the formulations and concepts described by Rivet et al. (2015), Roeloffs (1988), Talwani et al. (2007), and Tsai (2011). The transfer-function and correlation-analysis modules are still under development.

Please report any bugs you encounter; pull requests for improvements are also welcome. Thanks!

## Features
- dv/v computation
- Pressure modeling (pore pressure loading and thermoelastic loading are now available)
- Pressure-to-dv/v mapping (*in prep*)
- Correlation analysis (*in prep*)

## Usage
1. Set up `config.toml`.
2. Run the workflow:
```bash
python main.py
```

## Configuration
The workflow is controlled through `config.toml`, which currently has three main parts.
First, choose the workflow stage to run. The available stages are `dvv_calculation` and `pressure_modeling`. Second, review the MSNoise settings, including the data-discovery window, dv/v calculation parameters, and filter definitions. Third, configure the pressure-modeling inputs and physical parameters.

Most options are documented directly in `config.toml`; please review the inline comments before running a new workflow.

## Input File Formats
Input files are configured in `config.toml`. Relative paths are resolved from the project root.

### Pressure modeling inputs
The pressure-modeling stage reads standardized CSV files with the following columns:

- `groundwater_csv_path`: `time`, `temperature`, `groundwater level [m a.s.l.]`
- `atmospheric_pressure_csv_path`: `time`, `atmosphere pressure`
- `snow_csv_path`: `time`, `snow depth`; optional: `snow cover`

Units:

- `time` must be parseable by pandas.
- `temperature` is in degrees Celsius.
- `groundwater level [m a.s.l.]` is in meters above sea level.
- `atmosphere pressure` must be in Pa.
- `snow depth` is snow thickness in meters. The snow density used to convert depth to load is set by `pressure_modeling.snow_density_kg_m3`.
- `snow cover`, when provided, is a percentage from 0 to 100.

Optional external pressure loadings can be supplied with `pressure_modeling.external_pressure_loading_csv_path`. This file must contain `time` and one or more columns named `<driver>_loading_pa`, for example `ocean_tide_loading_pa` or `hydl_ewh_loading_pa`. These values must already be surface load or pressure in Pa. Do not place displacement, strain, or gravity-change predictors in this file.

### Synthetic dv/v mapping external predictors
The synthetic dv/v mapping stage always reads predictors from `outputs/02-pressure-modeling/pore_pressure_output.csv`. It can also read additional external predictor CSV files listed in `synthetic_dvv_mapping.external_predictor_paths`.

Each external predictor CSV must contain:

- `datetime`
- one or more numeric predictor columns

Predictor columns should be named `<driver>_<component>`. For GFZ-style loading deformation predictors, supported components are `duEW`, `duNS`, `duV`, and `dg`. Example columns are:

- `ntal_cf_duV`
- `ntol_cf_duEW`
- `hydl_cf_dg`

The driver prefix must match an entry in `synthetic_dvv_mapping.drivers`, such as `ntal_cf`, `ntol_cf`, or `hydl_cf`. `duEW`, `duNS`, and `duV` are displacement components in meters; `dg` is gravity change in `1e-8 m/s2`. These predictors are used directly in the mapping stage and are not converted to pore pressure.

## Formulation
*This section is under development. Please refer to the articles listed in the References section for now.*

## References
<p style="padding-left: 2em; text-indent: -2em;">
Lecocq, T., C. Caudron, et F. Brenguier (2014), MSNoise, a Python Package for Monitoring Seismic Velocity Changes Using Ambient Seismic Noise, Seismological Research Letters, 85(3), 715‑726, https://doi.org/10.1785/0220130073.
</p>
<p style="padding-left: 2em; text-indent: -2em;">
Rivet, D., Brenguier, F., and Cappa, F. (2015). Improved detection of preeruptive seismic velocity drops at the Piton de La Fournaise volcano. <em>Geophysical Research Letters</em>, 42, 6332-6339. https://doi.org/10.1002/2015GL064835
</p>
<p style="padding-left: 2em; text-indent: -2em;">
Roeloffs, E. A. (1988). Fault stability changes induced beneath a reservoir with cyclic variations in water level. <em>Journal of Geophysical Research</em>, 93(B3), 2107-2124. https://doi.org/10.1029/JB093iB03p02107
</p>
<p style="padding-left: 2em; text-indent: -2em;">
Talwani, P., Chen, L., and Gahalaut, K. (2007). Seismogenic permeability, ks. <em>Journal of Geophysical Research: Solid Earth</em>, 112, B07309. https://doi.org/10.1029/2006JB004665
</p>
<p style="padding-left: 2em; text-indent: -2em;">
Tsai, V. C. (2011). A model for seasonal changes in GPS positions and seismic wave speeds due to thermoelastic and hydrologic variations. <em>Journal of Geophysical Research: Solid Earth</em>, 116, B04404. https://doi.org/10.1029/2010JB008156
</p>
