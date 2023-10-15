import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression

from src.static.models import Models
from src.utils.utils import (
    generate_evals,
    generate_residual_plot
)

def linear_regression(data_train_x: pd.DataFrame, 
                      data_test_x: pd.DataFrame,
                      data_train_y: pd.DataFrame,
                      data_test_y: pd.DataFrame,
                      do_residuals: bool = True) -> List[str]:
    """Linear regression model on weather data from NOAA."""

    # Fit model with data
    regr = LinearRegression()
    regr.fit(data_train_x, data_train_y)
    predictions = regr.predict(data_test_x)
    residuals = data_test_y.values - predictions

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(model=Models.LINEAR_REGRESSION,
                          predictions=predictions,
                          data_test_y=data_test_y)