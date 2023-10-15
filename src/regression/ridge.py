import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit)

from src.utils.utils import (
    generate_evals,
    generate_test_train,
    generate_residual_plot
)

def ridge_regression(target: str,
                     core_features: list,
                     data: pd.DataFrame,
                     do_residuals: bool = True) -> List[str]:
    """"Ridge regression model on weather data from NOAA."""

    # Generate param search for ridge regression using alphas
    param_grid = {'alpha': np.logspace(-4, 4, 100)}

    # Split up data into x and y values
    data_x = data[core_features].values
    data_y = data[[target]].values

    # Using time series split to be used in cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Apply grid search cross validation with time series split and ridge regression
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(data_x, data_y)

    # Generate training and testing data
    data_train, data_test = generate_test_train(data=data)
    data_train_x, data_train_y = data_train[core_features], data_train[target]
    data_test_x, data_test_y = data_test[core_features], data_test[target]

    # Get the best model and fit on training data
    best_model = grid_search.best_estimator_
    best_model.fit(data_train_x, data_train_y)

    # Once we find best model then we predict and get residuals
    predictions = best_model.predict(data_test_x)
    residuals = data_test_y - predictions

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(test='Ridge Regression',
                          target=target,
                          predictions=predictions,
                          data_test=data_test)