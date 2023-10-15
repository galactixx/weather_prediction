import pandas as pd
from typing import List
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.static.models import Models
from src.utils.utils import (
    generate_evals,
    generate_residual_plot
)

def xgboost_regression(data_train_x: pd.DataFrame, 
                       data_test_x: pd.DataFrame,
                       data_train_y: pd.DataFrame,
                       data_test_y: pd.DataFrame,
                       do_residuals: bool = True,
                       do_feature_importances: bool = True) -> List[str]:
    """XGBoost regression model on weather data from NOAA."""
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0, 0.01, 0.1]
    }
    eval_set = [(data_test_x, data_test_y)]

    # Using time series split to be used in cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Apply grid search cross validation with time series split and xgboost regressor
    model = XGBRegressor(early_stopping_rounds=10, verbose=1, random_state=7)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error')
    grid_search.fit(data_train_x, data_train_y, eval_metric="rmse", eval_set=eval_set, verbose=True)

    # Get best parameter values from grid search
    best_params = grid_search.best_params_
    best_model = XGBRegressor(**best_params)
    best_model.fit(data_train_x, data_train_y)
    predictions = best_model.predict(data_test_x)
    residuals = data_test_y.values - predictions

    # Generate feature importances
    if do_feature_importances:
        importances = best_model.feature_importances_
        print(importances)

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(model=Models.XGBOOST_REGRESSION,
                          predictions=predictions,
                          data_test_y=data_test_y)