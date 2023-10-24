from typing import Optional, List

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit)

from src.static.models import Models
from src.utils.utils import (
    Data,
    generate_evals,
    generate_residual_plot
)

def xgboost_regression(data_test_train: Data,
                       do_residuals: bool = True,
                       do_feature_importances: bool = True,
                       hyperparameters: Optional[dict] = None) -> List[str]:
    """
    XGBoost regression model on weather data from NOAA.

    Parameters:
    - data_test_train: Data
        Data object containing training and testing data.
    - hyperparameters: dict or None
        A dictionary of XGBoost hyperparameters. If None, use default hyperparameters.
    ...
    """
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        }
    eval_set = [(data_test_train.x_val, data_test_train.y_val)]

    # Using time series split to be used in cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Apply randomized search cross-validation with time series split and XGBoost Regressor
    model = XGBRegressor(early_stopping_rounds=10, verbose=1, random_state=7)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparameters,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(data_test_train.x_train,
                      data_test_train.y_train,
                      eval_metric="rmse",
                      eval_set=eval_set,
                      verbose=True)

    # Get best parameter values from grid search
    best_params = random_search.best_params_
    best_model = XGBRegressor(**best_params)
    best_model.fit(data_test_train.x_train, data_test_train.y_train)
    predictions = best_model.predict(data_test_train.x_test)
    residuals = data_test_train.y_test.values - predictions

    # Generate feature importances
    if do_feature_importances:

        # Get feature importances as a dictionary
        importances = best_model.get_booster().get_fscore()

        # Create a DataFrame from the importances dictionary
        importance_df = pd.DataFrame(importances.items(), columns=['Feature', 'Importance'])

        # Sort the DataFrame by importance in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Reset the index of the DataFrame for a clean index
        importance_df = importance_df.reset_index(drop=True)
        print(importance_df)

    # Generate residuals and plot
    if do_residuals:
        generate_residual_plot(residuals=residuals,
                               predictions=predictions)

    # Generate all evals
    return generate_evals(model=Models.XGBOOST_REGRESSION,
                          predictions=predictions,
                          data_test_y=data_test_train.y_test)