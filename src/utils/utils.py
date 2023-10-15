import numpy as np
import pandas as pd
from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

from datetime import timedelta

def calculate_percentage_between_dates(percentage: float,
                                       start_date: pd.Timestamp,
                                       end_date: pd.Timestamp) -> pd.Timestamp:
    """Calculate date between start and end date which is some percentage between range of dates."""

    # Calculate the total time duration between start_date and end_date
    total_duration = (end_date - start_date).total_seconds()

    # Calculate the time duration for the given percentage
    target_duration = (percentage / 100) * total_duration

    # Calculate the target date by adding the duration to the start date
    target_date = start_date + timedelta(seconds=target_duration)
    return target_date

def generate_residual_plot(residuals: np.ndarray, predictions: np.ndarray) -> None:
    """generate residual plot and histogram after fitting model."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(predictions, residuals)
    axs[0].set_title('Residual Plot')

    axs[1].hist(residuals, bins=20, alpha=0.2)
    axs[1].set_title('Residual Histogram')

    plt.tight_layout()
    plt.show()

def generate_test_train(data: pd.DataFrame, percentage: float = 75.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test set from time series data given a percentage."""
    split_date = calculate_percentage_between_dates(percentage=percentage,
                                                    start_date=min(data.index),
                                                    end_date=max(data.index))

    data_train = data[data.index < split_date]
    data_test = data[data.index >= split_date]
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
