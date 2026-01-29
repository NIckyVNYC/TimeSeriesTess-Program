# TESS Light-Curve Anomaly Detection (Orion TadGAN)

This repository contains a Python pipeline for **detecting anomalies in TESS SPOC light curves** using an
**Orion TadGAN** model. It scans TESS sectors, runs anomaly detection per file, and produces **PDF reports**
and **CSV logs** for targets with high-severity anomalies.

> Core scripts (cleaned + commented):
> - `main_cleaned.py` — end-to-end training (optional) + detection + report generation
> - `library_cleaned.py` — shared imports + (legacy) training data load
> - `symFunc_cleaned.py` — pulse/transit symmetry heuristics

---

## What it does

1. **(Optional) Train/Fit TadGAN model** on one sample light curve (as written, it is configured to stop after the first file encountered in a single sector).
2. **Run detection** across a range of sectors (default: sectors 6–26).
3. If **any anomaly severity exceeds a threshold** (default: `0.90`), it:
   - Generates a **multi-page PDF** report for that target/sector with:
     - Raw and processed light curves
     - Background + centroid curves
     - Zoom-in plots for anomaly windows
     - A highlighted full-curve overlay for each anomaly window
   - Appends a row to a **master anomaly CSV** including:
     - TIC + sector
     - MAST TIC metadata (`Tmag`, `Vmag`, parallax, lum. class) when available
     - SIMBAD radial velocity (`RV_VALUE`) when available
     - Anomaly window start/end/severity arrays
     - A simple classification (`PULSE`, `TRANSIT`, or `NY-Classified`)
     - A symmetry label (`symmetry`, `no symmetry`, or `NA`)

---

## Requirements

### Python
- Python 3.9+ recommended

### Packages
You will need (at least):
- `numpy`, `pandas`, `matplotlib`
- `tensorflow` (environment dependency for many TadGAN setups)
- `orion` (Orion ML library / TadGAN pipeline)
- `astropy`, `lightkurve`
- `astroquery` (for MAST + SIMBAD queries)

> Note: package names and install steps can vary depending on your cluster/OS (especially Orion/TensorFlow).
> If you already run this on an HPC environment, prefer your existing module/conda setup.

---

## Data layout assumptions

The pipeline assumes your **TESS SPOC raw light curves** are stored by sector in folders like:

- `.../sector-06/`
- `.../sector-07/`
- ...
- `.../sector-26/`

The code uses two prefixes:
- `<prefix for <10>`: `.../sector-0`
- `<prefix for >=10>`: `.../sector-`

These are configured at the top of `main_cleaned.py`.

---

## Configuration

Open `main_cleaned.py` and edit these constants as needed:

- `MODEL_PATH` — where the trained model pickle is loaded/saved
- `SECTOR_PATH_PREFIX_LT10`, `SECTOR_PATH_PREFIX_GE10` — input data roots
- `TRAIN_TIMES_CSV`, `DETECT_TIMES_CSV`, `MASTER_ANOMALY_CSV` — output logs
- `SEVERITY_TRIGGER` — severity threshold that triggers report generation
- `PULSE_HIGH_THRESH`, `PULSE_LOW_THRESH`, `COUNT_THRESHOLD` — heuristic classifier thresholds

---

## How to run

From the repo directory:

```bash
python main_cleaned.py
```

### Notes on training
The “training” block is currently written to **break immediately** after encountering the first file in the walk,
meaning it will usually **not actually fit** unless you adjust the break logic.

If you want to fit on more data, remove/modify the `break` condition in the training loop.

---

## Outputs

### PDFs
For each target/sector with a triggered anomaly, a PDF is created:

```
TIC<id>_<sector>_<class>_<symmetry>.pdf
```

Example:
```
TIC123456789_6_TRANSIT_symmetry.pdf
```

### CSV logs
- `time_list_train.csv` — per-fit timing info
- `time_list_detect.csv` — per-file detection timing info
- `master_anomaly_list.csv` — master anomaly summary (one row per flagged file)

---

## File overview

### `main_cleaned.py`
- Loads or fits an Orion TadGAN model (pickle-based)
- Walks sector directories and runs detection per SPOC light curve
- Generates PDFs and writes CSV summaries when severity exceeds the threshold
- Uses:
  - `funcsTess.data_fix*` to ingest/clean light curve data into the expected format
  - `symFunc_cleaned.py` symmetry heuristics for quick labeling

### `library_cleaned.py`
- Centralizes imports used across the project
- Loads an Orion training signal (`S-1-train`) at import time (legacy behavior)
  - Consider making this a function if import-time work becomes a problem.

### `symFunc_cleaned.py`
- Implements symmetry heuristics for:
  - Pulse-like events (`sym_func_p`)
  - Transit-like events (`sym_func_t`)
- Splits the series into quarters and compares medians of extreme values (top-N / bottom-N)
- Includes a `verbose` flag (default True) to preserve original print-heavy behavior.

---

## Troubleshooting

- **“File not found” / empty scan**  
  Check `SECTOR_PATH_PREFIX_*` and confirm sector directories exist and are readable.

- **Orion/TadGAN import or runtime errors**  
  Ensure your Orion installation matches the expected API (and TensorFlow versions are compatible).

- **MAST/SIMBAD fields show `fail`**  
  Network access may be restricted (common on clusters), or the object query may not resolve.
  The pipeline is designed to keep going even if enrichment fails.

- **PDFs missing the first plot**  
  The script creates the first “raw” plot but does not save it to the PDF (this matches original behavior).
  If you want it included, add `pdf.savefig()` before closing that first figure.

---

## License / attribution

If you plan to publish this publicly, add your preferred license (MIT, Apache-2.0, etc.) and consider noting:
- Data source: NASA TESS SPOC light curves
- Catalog services: MAST TIC and SIMBAD (queried via `astroquery`)
- Anomaly model: Orion TadGAN

---

## Quick next improvements (optional)

- Add CLI args (e.g., `--sector-start`, `--sector-end`, `--severity`, `--model-path`)
- Avoid re-loading the model for every sector (load once unless you need isolation)
- Make training optional via a flag and train on a defined dataset split
- Replace filename parsing with a robust TIC/sector extraction method that does not rely on fixed `split("-")` indices
