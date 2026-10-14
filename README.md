# Kelmarsh Wind Turbine Health Monitoring

A data-driven condition-monitoring project using real-world SCADA and turbine status data from the Kelmarsh wind farm.

## Project Objective

Build a reproducible pipeline that cleans and validates turbine telemetry, investigates operational events, and develops methods to identify abnormal turbine behaviour and potential performance losses.

The goal is to support maintenance prioritisation with evidence-based indicators rather than making unsupported failure predictions.

## Dataset

The project uses the Kelmarsh wind farm dataset, including 10-minute SCADA measurements and turbine status/event records.

Source: [Kelmarsh Wind Farm Dataset — Zenodo](https://zenodo.org/records/16807551)

Raw and intermediate datasets are excluded from this repository. Download the dataset from its original source to reproduce the analysis.

## Work Completed

- Inspected raw SCADA files and their structure.
- Assessed missing values, duplicate timestamps, and signal availability.
- Built an interim dataset across multiple years.
- Investigated turbine status events and their relationship with power production.
- Analysed gearbox oil pressure patterns and a potential early-warning signal.
- Documented data-quality limitations and cases requiring further validation.

## Planned Work

- Validate candidate condition indicators against later periods and independent events.
- Develop expected-power and component-temperature baselines.
- Compare interpretable anomaly detection with a deep-learning approach where justified.
- Quantify potential energy losses and create transparent turbine-priority scores.
- Build an interactive Power BI dashboard.

## Important Limitations

A change in a sensor signal is not, by itself, proof of a component fault. Candidate indicators require out-of-time validation, and operational status records do not necessarily establish root cause.

## Repository Structure

- `scripts/` — data inspection, quality checks, preprocessing, and investigations.
- `reports/` — compact CSV outputs from exploratory analyses.
- `data/` — local datasets, excluded from version control.
- `models/` — trained models and related artefacts.
- `notebooks/` — exploratory analysis.
- `powerbi/` — dashboard files and supporting materials.
- `src/` — reusable project code.

## Reproducibility

Create a Python virtual environment, install dependencies from `requirements.txt`, download the source dataset, and place the files in the expected local data directories before running the scripts.

The project is under active development; modelling results and dashboard deliverables will be added as they are validated.