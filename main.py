'''
TESS light-curve anomaly detection pipeline (Orion/TadGAN).

This script:
  1) Optionally fits (trains) an Orion TadGAN model on a sample FITS light curve.
  2) Runs anomaly detection across TESS sectors, generating PDFs + CSV logs
     for any files with high-severity anomalies.
  3) Enriches results with MAST TIC + SIMBAD metadata where available.

Notes:
- Many imports and commented blocks reflect experimentation/iteration.
- Paths are currently hard-coded; consider moving them to constants or CLI args.
'''
# =========================
# Imports
# =========================

# Standard library
import csv
import os
import pickle
import time as time_now
import traceback
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import tensorflow as tf  # imported for environment/model deps (may be unused here)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Astronomy / TESS tooling
# (Some of these are only used in optional / commented sections. Keeping them
# makes it easier to toggle features back on without hunting imports.)
import astropy  # noqa: F401
import astropy.units as u  # noqa: F401
import lightkurve as lk  # noqa: F401
from astropy.io import fits  # noqa: F401
from astropy.timeseries import TimeSeries  # noqa: F401
from astropy.utils.data import get_pkg_data_filename  # noqa: F401
from astropy.wcs import WCS  # noqa: F401
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from lightkurve import TessLightCurveFile  # noqa: F401

# Orion
from orion.data import load_signal

# Local modules
import funcsTess
import symFunc


# =========================
# Convenience aliases
# =========================
data_fix = funcsTess.data_fix
data_fix2 = funcsTess.data_fix2
data_fix3 = funcsTess.data_fix3
sym_func_p = symFunc.sym_func_p
sym_func_t = symFunc.sym_func_t


# =========================
# Configuration
# =========================

# Model path (pickle) used for load/save
MODEL_PATH = Path("/home/nvasilescunyc/tess/TADGAN3/trained_model.pickle")

# TESS SPOC raw light curve directories (per-sector folders)
SECTOR_PATH_PREFIX_LT10 = "/data/scratch/data/tess/lcur/spoc/raws/sector-0"
SECTOR_PATH_PREFIX_GE10 = "/data/scratch/data/tess/lcur/spoc/raws/sector-"

# Logging outputs
TRAIN_TIMES_CSV = Path("time_list_train.csv")
DETECT_TIMES_CSV = Path("time_list_detect.csv")
MASTER_ANOMALY_CSV = Path("master_anomaly_list.csv")

# Detection threshold used to trigger report generation / catalog enrichment
SEVERITY_TRIGGER = 0.90

# Light-curve heuristics used for the simple classification logic below
PULSE_HIGH_THRESH = 1.05
PULSE_LOW_THRESH = 0.95
COUNT_THRESHOLD = 20


# =========================
# Optional / legacy data load
# =========================
# These are loaded but not used directly below; they may have been part of
# earlier experiments or are used indirectly in imported modules.
train_data = load_signal("S-1-train")
train_data2 = pd.DataFrame(data=train_data)


# =========================
# Helper functions
# =========================
def now_string() -> str:
    """Return a human-readable timestamp."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def load_model(path: Path):
    """Load a pickled Orion model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path: Path) -> None:
    """Save an Orion model to disk (pickle)."""
    with open(path, "wb") as f:
        pickle.dump(model, f)

    # Some Orion models support .save(); keep for compatibility.
    try:
        model.save(str(path))
    except Exception:
        pass


def append_csv_row(csv_path: Path, row: list) -> None:
    """Append a single row to a CSV file."""
    with open(csv_path, "a", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(row)


def write_master_header_if_needed(csv_path: Path) -> None:
    """Write the master anomaly CSV header if file is missing/empty."""
    header = [
        "TIC",
        "SECTOR",
        "Tmag",
        "Vmag",
        "Plx",
        "Lumclass",
        "RV_Value",
        "Start",
        "End",
        "Severity",
        "Date",
        "Classification",
        "Symmetry",
    ]

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        append_csv_row(csv_path, header)


def parse_tic_and_sector(file_address: str) -> tuple[str, str]:
    """Parse TIC ID + sector from the SPOC filename convention."""
    parts = file_address.split("-")

    # This mirrors the original logic; it will throw if the filename pattern changes.
    file_name = str(parts[3]).lstrip("0")
    sector = str(parts[2]).lstrip("s0")

    tic = f"TIC{file_name}"
    return tic, sector


def query_mast_tic(tic: str) -> dict:
    """Query MAST TIC catalog for basic fields (best-effort)."""
    try:
        catalog_data = Catalogs.query_object(tic, catalog="TIC")
        return {
            "Tmag": str(catalog_data[:1]["Tmag"]),
            "Vmag": str(catalog_data[:1]["Vmag"]),
            "plx": str(catalog_data[:1]["plx"]),
            "lumclass": str(catalog_data[:1]["lumclass"]),
        }
    except Exception:
        return {"Tmag": "fail", "Vmag": "fail", "plx": "fail", "lumclass": "fail"}


def query_simbad_rv(tic: str) -> str:
    """Query SIMBAD for RV_VALUE (radial velocity), best-effort."""
    try:
        Simbad.add_votable_fields("typed_id", "rv_value", "sp")
        result_table = Simbad.query_object(tic)
        return str(result_table["RV_VALUE"])
    except Exception:
        return "fail"


def clean_catalog_field(raw: str, remove_tokens: list[str]) -> str:
    """Strip common tokens from catalog-field strings."""
    cleaned = raw
    for tok in remove_tokens:
        cleaned = cleaned.replace(tok, "")
    return cleaned


def classify_lightcurve(second_col: np.ndarray) -> str:
    """Naive classification heuristic based on thresholded flux counts."""
    high_count = np.count_nonzero(second_col > PULSE_HIGH_THRESH, axis=0)
    low_count = np.count_nonzero(second_col < PULSE_LOW_THRESH, axis=0)

    if high_count > COUNT_THRESHOLD and low_count < COUNT_THRESHOLD:
        return "PULSE"
    if high_count < COUNT_THRESHOLD and low_count > COUNT_THRESHOLD:
        return "TRANSIT"
    return "NY-Classified"


def symmetry_classification(classify: str, second_col: np.ndarray) -> str:
    """Run the appropriate symmetry function based on the class."""
    if classify == "PULSE":
        return sym_func_p(second_col, 10)
    if classify == "TRANSIT":
        return sym_func_t(second_col, 10)
    return "NA"


# =========================
# Main script state
# =========================

# Placeholder from the original script (not used later). Kept so behavior stays similar.
legacy_file = open("test_file", "w")
legacy_writer = csv.writer(legacy_file)

times_fit = 0
times_detect = 0
n = 0  # total files encountered (across sectors)


# =========================
# (Optional) Fit / train loop
# =========================
# This loop currently trains on the FIRST file encountered in sector 1 only.
# Note: because of the `count > 0: break`, it runs at most once per sector.

orion_test = load_model(MODEL_PATH)

for y in range(1, 2):
    # Keep the original printed path behavior
    path_prefix = SECTOR_PATH_PREFIX_LT10 if y < 10 else SECTOR_PATH_PREFIX_GE10
    print(path_prefix)

    count = 0
    for root, dirs, files in os.walk(f"{path_prefix}{y}"):
        for name in files:
            count += 1

            # Break after the first file so you don't train on the whole sector.
            if count > 0:
                break

            start = time_now.time()
            file_address = f"{root}/{name}"

            # Fit the model on cleaned data from this file.
            orion_test.fit(data_fix(file_address))

            train_time = time_now.time() - start
            times_fit += 1

            print("this is times fit #", times_fit)
            print("This is name of file", name)

            append_csv_row(TRAIN_TIMES_CSV, [times_fit, train_time, now_string()])
            save_model(orion_test, MODEL_PATH)


# =========================
# Detection / reporting loop
# =========================

write_master_header_if_needed(MASTER_ANOMALY_CSV)

for y in range(6, 27):
    # Reload model each sector (matches original behavior)
    orion_test = load_model(MODEL_PATH)

    base_path = SECTOR_PATH_PREFIX_LT10 if y < 10 else SECTOR_PATH_PREFIX_GE10
    count = 0

    for root, dirs, files in os.walk(f"{base_path}{y}"):
        for name in files:
            x_zoom = []
            y_zoom = []

            n += 1
            count += 1
            times_detect += 1

            print("this is times detect #", times_detect)
            print("This is name of file", name)

            start2 = time_now.time()
            file_address = f"{root}/{name}"

            # Clean the data into Orion's expected input format
            open_file = data_fix(file_address)

            # Run anomaly detection
            anomalies = orion_test.detect(open_file)
            print(f"This is anomalies in file {file_address}:\n", anomalies)

            detect_time = time_now.time() - start2
            append_csv_row(DETECT_TIMES_CSV, [count, times_fit, detect_time, now_string()])

            # If there is at least one anomaly above the trigger, generate outputs
            if not anomalies[anomalies["severity"] > SEVERITY_TRIGGER].empty:
                print("success")

                cut_one_array = np.array(open_file)
                second_col = cut_one_array[:, [1]]

                classify = classify_lightcurve(second_col)
                sym_class = symmetry_classification(classify, second_col)

                # Arrays for plotting
                data_plot = np.array(data_fix2(file_address))
                data_plot2 = np.array(data_fix3(file_address))

                # Extract anomaly arrays (kept as arrays like original)
                start_arr = np.array(anomalies["start"])
                end_arr = np.array(anomalies["end"])
                severity_arr = np.array(anomalies["severity"])

                # Parse TIC + sector from filename/path
                try:
                    tic, sector = parse_tic_and_sector(file_address)
                except Exception:
                    print("Failed parsing TIC/sector from filename.")
                    traceback.print_exc()
                    continue

                # Catalog enrichment (best-effort)
                mast = query_mast_tic(tic)
                simb_rv = query_simbad_rv(tic)

                # Coordinates
                x_coordinate = data_plot[:, [0]]
                y_coordinate = data_plot[:, [1]]
                x2_coordinate = data_plot2[:, [0]]
                y2_coordinate = data_plot2[:, [1]]

                back_coordinate = data_plot[:, [2]]
                cent1_coordinate = data_plot[:, [3]]
                cent2_coordinate = data_plot[:, [4]]

                pdf_name = f"{tic}_{y}_{classify}_{sym_class}.pdf"

                with PdfPages(pdf_name) as pdf:
                    # Plot 1 (created but not saved in original)
                    plt.figure(figsize=(10, 5.2))
                    plt.title(f"{tic} Sector {y}")
                    plt.plot(x_coordinate, y_coordinate)
                    plt.xlabel("Time")
                    plt.ylabel("Light Values (Unchanged)")
                    plt.close()

                    # Plot 2 (saved)
                    plt.figure(figsize=(10, 5.2))
                    plt.title(f"Target {tic} Sector {y} {sym_class}")
                    plt.plot(x2_coordinate, y2_coordinate)
                    plt.xlabel("Time")
                    plt.ylabel("Relative Flux (Electrons Per Second)")
                    pdf.savefig()
                    plt.close()

                    # Plot 3: background
                    plt.figure(figsize=(10, 5.2))
                    plt.title(f"{tic} Sector {y} — Background curve")
                    plt.plot(x_coordinate, back_coordinate)
                    plt.xlabel("Time")
                    plt.ylabel("Flux (Electrons Per Second)")
                    pdf.savefig()
                    plt.close()

                    # Plot 4: centroid1
                    plt.figure(figsize=(10, 5.2))
                    plt.title(f"{tic} Sector {y} — Centroid 1 Curve")
                    plt.plot(x_coordinate, cent1_coordinate)
                    plt.xlabel("Time")
                    plt.ylabel("Flux (Electrons Per Second)")
                    pdf.savefig()
                    plt.close()

                    # Plot 5: centroid2
                    plt.figure(figsize=(10, 5.2))
                    plt.title(f"{tic} Sector {y} — Centroid 2 Curve")
                    plt.plot(x_coordinate, cent2_coordinate)
                    plt.xlabel("Time")
                    plt.ylabel("Flux (Electrons Per Second)")
                    pdf.savefig()
                    plt.close()

                    # Zoom + highlight plots for each detected anomaly window
                    array_anomalies = np.array(anomalies)

                    for row in array_anomalies:
                        # row is typically [start, end, severity, ...]
                        if row[2] <= 0.30:
                            continue

                        sever_no = float(row[2])
                        sever_str = f"{sever_no:.3f}"

                        # Expand window like original
                        start_idx = int(row[0] - 30) if row[0] > 30 else int(row[0])
                        end_idx = int(row[1])

                        # Build zoom arrays
                        x_zoom.clear()
                        y_zoom.clear()
                        for p in data_plot:
                            if start_idx < p[0] < end_idx:
                                x_zoom.append(p[0])
                                y_zoom.append(p[1])

                        # Zoom plot
                        plt.figure(figsize=(10, 5.2))
                        plt.title(f"Zoom severity {sever_str}")
                        plt.plot(x_zoom, y_zoom)
                        plt.xlabel("Time")
                        plt.ylabel("Flux (RAW)")
                        pdf.savefig()
                        plt.close()

                        # Full curve with highlighted segment
                        plt.figure(figsize=(10, 5.2))
                        plt.title(f"{tic} Sector {y} Severity from {start_idx} to {end_idx}")
                        plt.plot(x_coordinate, y_coordinate)

                        x2 = list(range(start_idx, end_idx))
                        if x2:
                            y_baseline = [y_coordinate[start_idx]] * len(x2)
                            plt.plot(x2, y_baseline)

                            # Arrow pointing down to the highlighted section
                            plt.arrow(
                                x2[0],
                                y_baseline[0] * 1.01,
                                0.0,
                                -0.04,
                                width=100,
                                head_width=300,
                                head_length=100,
                            )

                        plt.xlabel("Time")
                        plt.ylabel("Flux (RAW)")
                        pdf.savefig()
                        plt.close()

                # Append metadata to the master anomaly list CSV
                Tmag = clean_catalog_field(mast["Tmag"], ["Tmag", "-", "[", "]"])
                Vmag = clean_catalog_field(mast["Vmag"], ["Vmag", "-", "[", "]"])
                Plx = clean_catalog_field(mast["plx"], ["plx", "-", "[", "]"])
                LumD = clean_catalog_field(mast["lumclass"], ["lumclass", "-", "[", "]"])

                rvD = clean_catalog_field(simb_rv, ["RV_VALUE", "km", "/", "s", "-", "[", "]"])

                append_csv_row(
                    MASTER_ANOMALY_CSV,
                    [
                        tic.replace("TIC", ""),  # original wrote numeric file_name
                        sector,
                        Tmag,
                        Vmag,
                        Plx,
                        LumD,
                        rvD,
                        start_arr,
                        end_arr,
                        severity_arr,
                        now_string(),
                        classify,
                        sym_class,
                    ],
                )


# Final save (kept from original; note this is a different path than MODEL_PATH)
with open("/home/nvasilescunyc/tess/TADGAN/mypickle.pickle", "wb") as f:
    pickle.dump(orion_test, f)
