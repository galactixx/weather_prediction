from typing import Optional, Tuple, List

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
    x_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.DataFrame] = None

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
                        train_percentage: float = 0.70, 
                        do_validation: bool = False) -> Data:
    """Generate train and test set from time series data given a percentage."""

    def data_split_into_x_y(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Given some data, split into x and y datasets."""
        return data[core_features], data[target]
    
    # Determine the split points based on your data's time index
    test_percentage = 1 - train_percentage
    train_size = int(train_percentage * len(data))
    test_size = int(test_percentage * len(data))

    # Split the fataFrame into training, validation, and test sets
    data_train = data.iloc[:train_size]

    # Split the data accordingly into X and y for training set
    X_train, y_train = data_split_into_x_y(data=data_train)

    # Generate validation set only if set to do validation
    if do_validation:
        test_size = int((test_percentage / 2) * len(data))

        data_val = data.iloc[train_size:train_size + test_size]
        data_test = data.iloc[train_size + test_size:]

        X_test, y_test = data_split_into_x_y(data=data_test)
        X_val, y_val = data_split_into_x_y(data=data_val)

        return Data(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, x_val=X_val, y_val=y_val)
    else:
        data_test = data.iloc[train_size:]
        X_test, y_test = data_split_into_x_y(data=data_test)

        return Data(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)

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
