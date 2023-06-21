"""Provides common routines used in the models implemented in this directory"""
from typing import List

import numpy as np
import pandas as pd


def remove_outliers(
    df0: pd.DataFrame, columns: List = None, verbose: bool = False, outlier: float = 3.0
) -> pd.DataFrame:
    """Removes rows where any column has data greater than 3 standard deviations
    from the mean. If 'columns' are specified, only include these in the
    calculation."""
    df_include = df0 if columns is None else df0[columns]
    df_reduced = df0[
        ((np.abs(df_include - df_include.mean()) / df_include.std()) < outlier).all(
            axis=1
        )
    ]
    nrm = len(df0) - len(df_reduced)
    if verbose:
        print(f"Removed {nrm} rows ({100*nrm/len(df0):.1f}%) marked as outliers.")
    return df_reduced

def get_number_of_parameters(model):
    return np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])