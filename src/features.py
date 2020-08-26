# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

base_in = Path("data", "clean")
base_out = Path("data", "featured")

base_out.mkdir(exist_ok=True)

data_train = pd.read_csv(Path(base_in, "input_bcell_sars.csv"))
data_test = pd.read_csv(Path(base_in, "input_covid.csv"))

# Add protein and peptide length features
# To get a quick and dirt pipeline running, let's go numeric features only
data_train.insert(5, "protein_len", data_train["protein_seq"].map(len))
data_train.insert(5, "peptide_len", data_train["peptide_seq"].map(len))
data_train.iloc[:, 5:].to_csv(Path(base_out, "training.csv"), index=False)

data_test.insert(5, "protein_len", data_test["protein_seq"].map(len))
data_test.insert(5, "peptide_len", data_test["peptide_seq"].map(len))
data_test.iloc[:, 5:].to_csv(Path(base_out, "testing.csv"), index=False)
