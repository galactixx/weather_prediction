import numpy as np
import pandas as pd
from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

def generate_residual_plot(residuals: np.ndarray, predictions: np.ndarray) -> None:
    """generate residual plot and histogram after fitting model."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(predictions, residuals)
    axs[0].set_title('Residual Plot')

    axs[1].hist(residuals, bins=20, alpha=0.2)
    axs[1].set_title('Residual Histogram')

    plt.tight_layout()
    plt.show()

def generate_test_train(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate train and test set from time series data. Data only goes from 1/1/2010 -> 9/24/2023"""
    data_train = data[data.index < pd.to_datetime('6/1/2017')]
    data_test = data[data.index >= pd.to_datetime('6/1/2017')]
    return data_train, data_test

def generate_evals(test: str, target: str, predictions: list, data_test: pd.DataFrame) -> List[str]:
    """generate evals after fitting and predicting values."""
    evals = []

    # get r-squared from test data
    r_squared = r2_score(data_test[target], predictions)
    evals.append(f'R-Squared ({test}): {r_squared}')

    # get the mae
    mae = mean_absolute_error(data_test[target], predictions)
    evals.append(f'Mean Absolute Error ({test}): {mae}')

    # get the mse
    mse = mean_squared_error(data_test[target], predictions)
    evals.append(f'Mean Squared Error ({test}): {mse}')
    return evals
