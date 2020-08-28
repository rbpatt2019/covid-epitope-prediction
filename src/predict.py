# -*- coding: utf-8 -*-
from pathlib import Path

import joblib

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yaml

with open(Path("params.yaml"), "r") as file:
    params = yaml.safe_load(file)
    seed = params["seed"]

np.random.seed(seed)

pl = joblib.load(Path("models", "trained.joblib"))

data = pd.read_csv(Path("data", "featured", "testing.csv"))

predictions = pl.predict(data.to_numpy())

joblib.dump(predictions, Path("results", "predictions.joblib"))
