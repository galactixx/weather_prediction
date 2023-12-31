import os
import json
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import chain
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.static.columns import NOAANames
from src.utils.utils import generate_test_train
from src.regression.ridge import ridge_regression
from src.regression.linear import linear_regression
from src.regression.xgboost import xgboost_regression

warnings.filterwarnings('ignore')

# From documentation: 'Note: 9’s in a field (e.g.9999) indicate missing data or data that has not been received'
# Remove these values as they are not valid to use in analysis
# --- NOT AN ISSUE WHEN OUR FILTERING IS APPLIED ---

# Interations with our data (between TMIN and TMAX) did not result in a better model
# No need for transformations of variables

with open('config.json') as config_file:
    config = json.load(config_file)

date_today = datetime.today().date()

# Path variables and other constants
BASE_PATH = config["BASE_PATH"]
EVALS_PATH = config["EVALS_PATH"]
ROLLING_MEAN_DAYS = config["ROLLING_MEAN_DAYS"]
DO_VIF_TEST = config["DO_VIF_TEST"]
DO_LINEAR_COMPARISON = config["DO_LINEAR_COMPARISON"]
DO_CORRELATION_MATRIX = config["DO_CORRELATION_MATRIX"]
DO_FEATURE_DISTRIBUTION = config["DO_FEATURE_DISTRIBUTION"]

TEMP_COLUMNS = [NOAANames.TMIN, NOAANames.TMAX]

contents = os.listdir(BASE_PATH)

def _generate_weights(days: int) -> list:
    """Dynamically generate weights based on number of days as input."""
    return list(range(1, days+1))

def _load_data(file: str) -> pd.DataFrame:
    """Load in individual csv file from data folder."""
    if file.endswith('.csv'):
        return pd.read_csv(f'{BASE_PATH}{file}', parse_dates=[NOAANames.DATE])
    else:
        raise Exception(f'only accepted files are csv, {file} is not a csv. please address')

def _generate_dummies(column: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Generate dummies for a given pandas series."""
    data_dummies = pd.get_dummies(data[column], prefix=column)
    data = data.drop(column, axis=1)
    return data.join(data_dummies), data_dummies.columns.tolist()

def _generate_feature_distributions(core_features: list, data_filtered: pd.DataFrame) -> None:
    """Generate qq-plot and histogram with features."""
    for feature in core_features:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        stats.probplot(data_filtered[feature], dist="norm", plot=axs[0])
        axs[0].set_title('Q-Q Plot')

        axs[1].hist(data_filtered[feature], bins=20, alpha=0.2)
        axs[1].set_title('Histogram')

        plt.tight_layout()
        plt.show()

def _generate_linear_comparisons(target: str, core_features: list, data_filtered: pd.DataFrame) -> None:
    """Check linear relationship between features and target."""
    for feature in core_features:
        plt.scatter(data_filtered[feature], data_filtered[target])
        plt.show()

def _generate_vif_test(core_features: list, data_filtered: pd.DataFrame) -> None:
    """Check for multicollinearity within data."""
    X = add_constant(data_filtered[core_features])
    vif_data = pd.Series([variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])], index=X.columns)
    print(vif_data)

def _generate_correlation_matrix(data_filtered: pd.DataFrame) -> None:
    """Correlation matrix."""
    matrix = data_filtered.corr().round(2)
    sns.heatmap(matrix, annot=True)
    plt.show()

if __name__ == '__main__':

    # Make index as date
    data = pd.concat(_load_data(file=file) for file in contents)
    data = data.sort_values(by=[NOAANames.DATE])
    data.index = data[NOAANames.DATE]

    # Shift TMAX value by one groupbed by STATION
    data[NOAANames.TMAX_NEXT_DAY] = data.groupby([NOAANames.STATION])[NOAANames.TMAX].shift(-1)

    core_features = [NOAANames.PRCP, NOAANames.STATION, NOAANames.TMIN, NOAANames.TMAX]
    # Since there are multiple stations in the data, we would generate categorical variable based on STATION
    # but since only one station actually has non-null TMAX values, we will not do this and instead
    # remove null station data
    data = data[[NOAANames.TMAX_NEXT_DAY] + core_features].dropna()
    stations = data[NOAANames.STATION].unique().tolist()

    def generate_additional_features(station: str, core_features: list, data: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features."""
        data_temp = data[data[NOAANames.STATION] == station]
        for col in TEMP_COLUMNS:
            for days in ROLLING_MEAN_DAYS:
                new_feature_column = f'{col}_{days}_DAY_WEIGHT_AVG'
                weights = _generate_weights(days=days)
                data_temp[new_feature_column] = data_temp[col].rolling(days).apply(
                    lambda x: np.dot(x, weights) / sum(weights), raw=True)
                if new_feature_column not in core_features:
                    core_features.append(new_feature_column)
        return data_temp
    data = pd.concat(generate_additional_features(station=station, core_features=core_features,
                                                  data=data) for station in stations)
    
    # Filter only for core features plus target
    data_filtered = data[[NOAANames.TMAX_NEXT_DAY] + core_features]

    # Remove all na values, while we could fill in the NULL percipitation values
    # Its a small % so we remove to facilitate analysis
    data_filtered = data_filtered.dropna()

    # Generate dummies for STATION variable
    data_filtered, dummy_columns = _generate_dummies(column=NOAANames.STATION, data=data_filtered)
    core_features.extend(dummy_columns)
    core_features.remove(NOAANames.STATION)

    # Check linear relationship between features and target
    if DO_LINEAR_COMPARISON:
        _generate_linear_comparisons(target=NOAANames.TMAX_NEXT_DAY,
                                     core_features=core_features,
                                     data_filtered=data_filtered)

    # Check for multicollinearity within data
    if DO_VIF_TEST:
        _generate_vif_test(core_features=core_features, data_filtered=data_filtered)

    # Correlation matrix
    if DO_CORRELATION_MATRIX:
        _generate_correlation_matrix(data_filtered=data_filtered)

    # Check normal distribution of features
    if DO_FEATURE_DISTRIBUTION:
        _generate_feature_distributions(core_features=core_features,
                                        data_filtered=data_filtered)

    # Generate training and testing data for linear/ridge regression
    data_test_train = generate_test_train(data=data_filtered,
                                          target=NOAANames.TMAX_NEXT_DAY,
                                          core_features=core_features)
    
    # Generate training and testing data for xgboost regression
    data_test_train_val = generate_test_train(data=data_filtered,
                                              target=NOAANames.TMAX_NEXT_DAY,
                                              core_features=core_features,
                                              do_validation=True)

    # All evals generated from regression model
    evals = [
        linear_regression(data_test_train=data_test_train),
        ridge_regression(data_test_train=data_test_train),
        xgboost_regression(data_test_train=data_test_train_val)
    ]
    evals = list(map(str, chain.from_iterable(evals)))
    with open(EVALS_PATH, 'w') as f:
        for eval in evals:
            f.write(f"{date_today} {eval}\n")
