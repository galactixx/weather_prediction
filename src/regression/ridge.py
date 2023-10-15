import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

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
    alphas = np.logspace(-4, 4, 100)
    data_x = data[core_features].values
    data_y = data[[target]].values
    
    return_alphas = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for alpha in alphas:
        for train_index, test_index in tscv.split(data_x):
            X_train, X_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            
            # fit model using alpha and predict values and calcualte mse
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            error = mean_squared_error(y_test, predictions)
            return_alphas.update({error: alpha})

    # find best model after cross-validation
    best_alpha = [return_alphas[i] for i in return_alphas if i == min(return_alphas)][0]

    # train model on training data
    data_train, data_test = generate_test_train(data=data)
    model = Ridge(alpha=best_alpha).fit(data_train[core_features], data_train[target])

    # once we find best model then we predict and get residuals
    predictions = model.predict(data_test[core_features])
    residuals = np.array(data_test[target]) - predictions

    # get predictions and generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # generate all evals
    return generate_evals(test='Ridge Regression',
                          target=target,
                          predictions=predictions,
                          data_test=data_test)