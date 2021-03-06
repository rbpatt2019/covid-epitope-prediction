# -*- coding: utf-8 -*-
"""The data provide are actually impressively clean - no missing values,
consistent d-types, consistent formatting, etc. The only immediately obviously
cleaning necessary is to join input_bcell.csv with input_sars.csv, as they are
both considered training data
"""
import shutil
from pathlib import Path

import pandas as pd

base_in = Path("data", "raw")
base_out = Path("data", "clean")

base_out.mkdir(exist_ok=True)

data_bcell = pd.read_csv(Path(base_in, "input_bcell.csv"))
data_sars = pd.read_csv(Path(base_in, "input_sars.csv"))

pd.concat([data_bcell, data_sars], axis=0).reset_index(drop=True).to_csv(
    Path(base_out, "input_bcell_sars.csv"), index=False
)

# Test data requires no cleaning, so it is simply copied to data/clean
shutil.copy(Path(base_in, "input_covid.csv"), Path(base_out, "input_covid.csv"))
