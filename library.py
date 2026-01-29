"""library.py

A shared import/utility module used by the TESS/Orion TadGAN pipeline.

This file mostly centralizes imports used across the project, and (as written)
also loads a small Orion training signal into `train_data` / `train_data2` at
import-time.

If you find imports are slow or you're importing this module in many places,
consider moving the training-data load into a function (lazy load) instead.
"""

# =========================
# Standard library
# =========================
import csv
import math
import os
import pickle
import time as time_now
import traceback
from datetime import datetime
from pathlib import Path

# =========================
# Third-party
# =========================
import numpy as np
import pandas as pd
import tensorflow as tf  # noqa: F401 (kept for environment/model deps)

import matplotlib.pyplot as plt  # noqa: F401
from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

# =========================
# Astronomy / TESS tooling
# =========================
import astropy  # noqa: F401
import astropy.units as u  # noqa: F401
import lightkurve as lk  # noqa: F401
from astropy.io import fits  # noqa: F401
from astropy.timeseries import TimeSeries  # noqa: F401
from astropy.utils.data import get_pkg_data_filename  # noqa: F401
from astropy.wcs import WCS  # noqa: F401
from astroquery.mast import Catalogs  # noqa: F401
from astroquery.simbad import Simbad  # noqa: F401
from lightkurve import TessLightCurveFile  # noqa: F401

# =========================
# Orion
# =========================
from orion.data import load_signal

# -----------------------------------------------------------------------------
# Training data (loaded at import time)
# -----------------------------------------------------------------------------
# NOTE: Importing this module will immediately load Orion's "S-1-train" dataset.
# That is convenient for notebooks, but can be surprising in production code.
# If you'd prefer lazy loading, wrap this in a function.
train_data = load_signal("S-1-train")
train_data2 = pd.DataFrame(data=train_data)
