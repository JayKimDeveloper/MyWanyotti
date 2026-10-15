#!/usr/bin/env python3
"""Validate the pinned local Python environment without training a model."""

from __future__ import annotations

import json
import platform

import catboost
import numpy
import pandas
import pyarrow
from catboost import CatBoostClassifier


def main() -> None:
    classifier = CatBoostClassifier(iterations=1, task_type="CPU", verbose=False)
    result = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "catboost": catboost.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "catboost_task_type": classifier.get_params()["task_type"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
