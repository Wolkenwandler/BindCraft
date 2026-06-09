import os, re, shutil, time, json, gc
import argparse
import pickle
import warnings
import zipfile
import numpy as np
import pandas as pd
import math, random
import matplotlib.pyplot as plt

# BINDCRAFT_MODE selects which heavy dependency stacks this process loads, so the
# codebase can run as a single all-in-one process (default) or split across images:
#   local   (default) -> PyRosetta + colabdesign/jax in one process (unchanged behaviour)
#   gpu               -> colabdesign/jax + Rosetta-over-HTTP client (no PyRosetta installed)
#   rosetta           -> PyRosetta only; colabdesign/jax is NOT imported
_MODE = os.environ.get("BINDCRAFT_MODE", "local")
if _MODE == "gpu":
    from .rosetta_client import *
else:
    from .pyrosetta_utils import *
if _MODE != "rosetta":
    from .colabdesign_utils import *
from .biopython_utils import *
from .generic_utils import *

# suppress warnings
#os.environ["SLURM_STEP_NODELIST"] = os.environ["SLURM_NODELIST"]
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=BiopythonWarning)