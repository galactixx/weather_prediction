import numpy as np
import pandas as pd
from typing import List
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.utils.utils import (
    generate_evals,
    generate_residual_plot
)

def xgboost_regression(data_train_x: pd.DataFrame, 
                       data_test_x: pd.DataFrame,
                       data_train_y: pd.DataFrame,
                       data_test_y: pd.DataFrame,
                       do_residuals: bool = True) -> List[str]:
    """XGBoost regression model on weather data from NOAA."""
    param_grid = {}

    # Fit model with data
    xgb = XGBRegressor(objective="reg:squarederror")
    xgb.fit(data_train_x, data_train_y)
    predictions = xgb.predict(data_test_x)
    residuals = data_test_y.values - predictions

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(test='XGBoost Regression',
                          predictions=predictions,
                          data_test_y=data_test_y)