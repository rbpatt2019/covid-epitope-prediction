# -*- coding: utf-8 -*-
from pathlib import Path

from joblib import dump
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

# Guick and dirty - GaussianNB with PowerTransformer
scl = PowerTransformer()
clf = GaussianNB()

pl = Pipeline([("Scale", scl), ("Classify", clf)])

# As GaussianNB has functionally no parameters, GridSearchCV isn't necessary
# So I dump the model and will do CV in evaluate.py
dump(pl, Path("models", "untrained.joblib"))
