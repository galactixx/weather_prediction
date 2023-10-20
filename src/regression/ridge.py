from typing import List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit)

from src.static.models import Models
from src.utils.utils import (
    Data,
    generate_evals,
    generate_residual_plot
)

def ridge_regression(data_test_train: Data,
                     do_residuals: bool = True) -> List[str]:
    """"Ridge regression model on weather data from NOAA."""

    # Generate param search for ridge regression using alphas
    param_grid = {'alpha': np.logspace(-4, 4, 100)}

    # Using time series split to be used in cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Apply grid search cross validation with time series split and ridge regression
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(data_test_train.x_train, data_test_train.y_train)

    # Get the best model and fit on training data
    best_model = grid_search.best_estimator_
    best_model.fit(data_test_train.x_train, data_test_train.y_train)

    # Once we find best model then we predict and get residuals
    predictions = best_model.predict(data_test_train.x_test)
    residuals = data_test_train.y_test.values - predictions

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(model=Models.RIDGE_REGRESSION,
                          predictions=predictions,
                          data_test_y=data_test_train.y_test)