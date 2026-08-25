# UAS Survey Tool v2.0

A Python desktop GIS application for terrain-aware UAS survey planning, AOI review, GCP/VCP layout generation, elevation-data handling, geospatial overlays, QA/QC, reporting, and field deliverables.

Built by SALT19 for operational use within MasTec survey and engineering workflows, then prepared for public open-source release with embedded credentials and local artifacts removed.

> **Scope:** Planning and decision-support software. Operators remain responsible for regulatory compliance, site authorization, airspace review, aircraft limitations, weather assessment, and safe mission execution.

## Capabilities

- Load survey areas of interest from KMZ/KML
- Generate configurable GCP and VCP layouts
- Add DEM/DSM terrain context
- Calculate terrain-aware planning values
- Review points and overlays before field execution
- Export CSV, KML, PDF, KMZ, and related survey deliverables
- Support repeatable QA/QC and reporting workflows

## Operational workflow

```text
KMZ/KML AOI → terrain context → GCP/VCP layout
            → review and QA/QC → field deliverables → validation
```

## Requirements

- Windows 10/11 recommended
- Python 3.10
- Conda or Miniconda recommended for GDAL/Rasterio/Fiona compatibility

## Setup

### Recommended: Conda

```powershell
conda env create -f environment.yaml
conda activate uas_survey_tool_v2
python bootstrap_main.py
```

### Alternative: pip

Use `requirements.txt` only after installing a Python environment that already has compatible native GIS libraries.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python bootstrap_main.py
```

## OpenTopography API setup

The app does not ship with a built-in OpenTopography key.

Choose one of these methods before using DEM/DSM auto-download:

1. Set `OPENTOPO_API_KEY` in the environment.
2. Copy [config.example.json](config.example.json) to `config.json` and insert your key.
3. Enter the key when prompted on the first DEM/DSM download.

```powershell
$env:OPENTOPO_API_KEY="your-api-key"
python bootstrap_main.py
```

OpenTopoData-based elevation fallback does not require an OpenTopography key. Never commit a real API key.

## Usage

1. Launch `python bootstrap_main.py`.
2. Load a KMZ/KML AOI.
3. Configure flight and layout settings.
4. Optionally enable DEM/DSM auto-download.
5. Generate and review the plan.
6. Export the required field and reporting deliverables.

## Validation

```powershell
python -m compileall .
python -m pytest tests -q
```

## Repository hygiene

The public repository excludes local logs, caches, packaged environments, build outputs, license artifacts, and credential files. See [.gitignore](.gitignore) and [SECURITY.md](SECURITY.md) before contributing.

## Related SALT19 systems

- [AeroClear — UAS flight intelligence and mission readiness](https://aeroclear.salt19.com/)
- [SALT19 portfolio](https://salt19.com/founder)
- [Contributing](CONTRIBUTING.md)

## License

[MIT](LICENSE) © 2026 Gabriel V Allit
