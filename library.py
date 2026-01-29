import numpy as np
import os
import pandas as pd
import csv
from numpy import ndarray
from datetime import datetime
#pd.set_option('display.max_rows', None)
import tensorflow as tf
import time as time_now

import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import lightkurve as lk
from lightkurve import TessLightCurveFile
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.timeseries import TimeSeries
import pickle
from pathlib import Path
from orion.data import load_signal
global train_data
train_data = load_signal('S-1-train')
global train_data2
train_data2 = pd.DataFrame(data = train_data)
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
import traceback
import math
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime

