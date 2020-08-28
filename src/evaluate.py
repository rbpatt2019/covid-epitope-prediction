# -*- coding: utf-8 -*-
import time
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
    seed = params["seed"]
    cv = params["cv"]

np.random.seed(seed)

data = pd.read_csv(Path("data", "featured", "training.csv"))
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()
del data

pl = joblib.load(Path("models", "untrained.joblib"))

# Will use StratifiedKFold and manual looping
# Builtin cross_validate functions are only easily usable with metrics returning
# Single values, not curves

train_time = []

precision = []
recall = []
fscore = []

precision_curve = []
recall_curve = []

tpr_roc = []
fpr_roc = []

skf = StratifiedKFold(n_splits=cv, shuffle=False)
for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    start = time.time()
    pl.fit(X_train, y_train)
    end = time.time()
    train_time.append(end - start)

    y_pred = pl.predict(X_test)
    y_prob = pl.predict_proba(X_test)[:, 1]

    prec, rcl, fsc, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    precision.append(prec)
    recall.append(rcl)
    fscore.append(fsc)

    prec_curve, rcl_curve, _ = precision_recall_curve(y_test, y_prob)
    precision_curve.append(prec_curve)
    recall_curve.append(rcl_curve)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tpr_roc.append(tpr)
    fpr_roc.append(fpr)

auc_score = [auc(x, y) for x, y in zip(fpr_roc, tpr_roc)]

# Now dump metrics
def float_representer(dumper, value):
    text = "{0:.4f}".format(value)
    return dumper.represent_scalar(u"tag:yaml.org,2002:float", text)


yaml.SafeDumper.add_representer(float, float_representer)

metrics = {
    "Mean Precision": mean(precision),
    "St.Dev. Precision": stdev(precision),
    "Mean Recall": mean(recall),
    "St.Dev. Recall": stdev(recall),
    "Mean F-score": mean(fscore),
    "St.Dev. F-score": stdev(fscore),
    "Mean AUC": mean(auc_score),
    "St.Dev. AUC": stdev(auc_score),
    "Mean Training Time": mean(train_time),
    "St.Dev. Training Time": stdev(train_time),
}
yaml.safe_dump(metrics, Path("results", "metrics.yaml"))

# And plots
pr_curve = pd.DataFrame(
    {"Recall": np.mean(recall_curve), "Precision": np.mean(precision_curve)}
)
pr_curve.to_csv(Path("results", "pr_curve.csv"), index=False)

roc_curve = pd.DataFrame(
    {"False Positive Rate": np.mean(fpr_roc), "True Positive Rate": np.mean(tpr_roc)}
)
roc_curve.to_csv(Path("results", "roc_curve.csv"), index=False)

# Finally, train on whole data and dump
pl.fit(X, y)
joblib.dump(pl, Path("models", "trained.joblib"))
