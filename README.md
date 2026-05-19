# Environmental-Seismic-Analysis-Workflow
Hong-Mao Huang, 2026
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
