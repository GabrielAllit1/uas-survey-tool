# UAS Survey Tool v2.0

Desktop GIS planning tool for loading an AOI from KMZ/KML, generating GCP/VCP layouts, downloading terrain data, and exporting survey deliverables.

## Release Notes

This repository has been prepared for open-source release by removing embedded API fallback behavior, documenting first-run configuration, and excluding local artifacts such as logs, caches, packaged environments, and license secret files from source control.

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

Use `requirements.txt` only after installing a Python environment that already has compatible native GIS libraries available.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python bootstrap_main.py
```

## OpenTopography API Setup

The app no longer ships with a built-in OpenTopography key.

Choose one of these options before using DEM/DSM auto-download:

1. Set an environment variable:

```powershell
$env:OPENTOPO_API_KEY="your-api-key"
python bootstrap_main.py
```

2. Create a config file from [config.example.json](config.example.json):

```powershell
Copy-Item config.example.json config.json
```

Then replace `YOUR_OPENTOPOGRAPHY_API_KEY` with your real key.

3. Launch the app and enter the key when prompted on first DEM/DSM download. The key will be stored in user settings and `~/.uas_survey_tool/config.json`.

OpenTopoData-based elevation fallback does not require an OpenTopography key.

## Usage

1. Launch `python bootstrap_main.py`.
2. Load a KMZ/KML AOI.
3. Configure flight and layout settings.
4. Optionally enable DEM/DSM auto-download.
5. Generate the plan.
6. Review points and export CSV, KML, PDF, and other deliverables.

## Tests

```powershell
python -m compileall .
python -m pytest tests -q
```

If `pytest` is not installed in the active interpreter, install it first or activate the project environment.

## Sensitive Files Not Intended For Source Control

- `secure_key.dat`
- `license_data.dat`
- `license_key_log.csv`
- `opentopography_api.txt`
- `logs/`
- `dem_cache/`
- `dsm_cache/`
- `build/`, `dist/`, `dist_release/`
- bundled local environment directory such as `uas_survey_tool_v2/`

## Packaging

- `UAS_Survey_Tool_v2.spec` and `bootstrap_main.spec` now resolve GIS runtime paths from the active environment instead of a hardcoded local machine path.
- `build_and_sign_windows.py` now defaults to the current project directory.

