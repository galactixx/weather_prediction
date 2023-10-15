import numpy as np
import pandas as pd
from typing import List
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.utils.utils import (
    generate_evals,
    generate_residual_plot
)

def xgboost_regression(target: str,
                       core_features: list,
                       data_test: pd.DataFrame,
                       data_train: pd.DataFrame,
                       do_residuals: bool = True) -> List[str]:
    """XGBoost regression model on weather data from NOAA."""

    # fit model with data
    xgb = XGBRegressor(objective="reg:squarederror")
    xgb.fit(data_train[core_features], data_train[target])
    predictions = xgb.predict(data_test[core_features])
    residuals = np.array(data_test[target]) - predictions

    # get predictions and generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # generate all evals
    return generate_evals(test='XGBoost Regression',
                          target=target,
                          predictions=predictions,
                          data_test=data_test)