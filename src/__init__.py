# -*- coding: utf-8 -*-
"""This folder contains the scripts necessary for replicating analyses.

They are:
    clean.py - Make the data clean (merge datasets, fill NAs, data formatting)
    features.py - Make features from given data
    model.py - Make and train a model
    evaulate.py - Make predictions and get metrics on performance

Reusable parametres are specified in params.yaml

All steps were executed using `dvc run`, so the analysis can be reproduced using
`dvc repro`
"""
