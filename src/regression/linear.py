from typing import List

from sklearn.linear_model import LinearRegression

from src.static.models import Models
from src.utils.utils import (
    Data,
    generate_evals,
    generate_residual_plot
)

def linear_regression(data_test_train: Data,
                      do_residuals: bool = True) -> List[str]:
    """Linear regression model on weather data from NOAA."""

    # Fit model with data
    regr = LinearRegression()
    regr.fit(data_test_train.x_train, data_test_train.y_train)
    predictions = regr.predict(data_test_train.x_test)
    residuals = data_test_train.y_test.values - predictions

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(model=Models.LINEAR_REGRESSION,
                          predictions=predictions,
                          data_test_y=data_test_train.y_test)