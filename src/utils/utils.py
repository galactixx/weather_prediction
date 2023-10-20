from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

from src.static.models import Models

@dataclass
class Data:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame

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

def generate_test_train(data: pd.DataFrame,
                        target: str,
                        core_features: list,
                        percentage: float = 75.0) -> Data:
    """Generate train and test set from time series data given a percentage."""
    split_date = calculate_percentage_between_dates(percentage=percentage,
                                                    start_date=min(data.index),
                                                    end_date=max(data.index))

    # Split up into initial train and test datasets
    data_train = data[data.index < split_date]
    data_test = data[data.index >= split_date]

    # Further split up the datasets into x and y
    data_train_x, data_train_y = data_train[core_features], data_train[target]
    data_test_x, data_test_y = data_test[core_features], data_test[target]

    return Data(x_train=data_train_x, x_test=data_test_x, y_train=data_train_y, y_test=data_test_y)

def generate_evals(model: Models, predictions: list, data_test_y: pd.DataFrame) -> List[str]:
    """Generate evals after fitting and predicting values."""
    evals = []

    # R-squared
    r_squared = r2_score(data_test_y, predictions)
    evals.append(f'R-Squared ({model}): {r_squared}')

    # MAE
    mae = mean_absolute_error(data_test_y, predictions)
    evals.append(f'Mean Absolute Error ({model}): {mae}')

    # MSE
    mse = mean_squared_error(data_test_y, predictions)
    evals.append(f'Mean Squared Error ({model}): {mse}')
    return evals
