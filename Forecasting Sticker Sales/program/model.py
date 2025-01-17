# Data manipulation
# ==============================================================================
import numpy as np

# Modeling and Forecasting
# ==============================================================================
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterSarimax
from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import grid_search_forecaster
from skforecast.exceptions import LongTrainingWarning

from skforecast.direct import ForecasterDirect
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.simplefilter('ignore', category=LongTrainingWarning)

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 10

y_column = ['num_sold']
steps = 562

class Forecast:

    def create_random_forest_regresor_forecaster(self, df, y_column, steps):
        forecaster = ForecasterRecursive(
                        regressor = RandomForestRegressor(random_state=123),
                        lags      = 6
                    )
        return forecaster
    
    def create_lgbm_regressor_forecaster(self, df, y_column, steps):
        forecaster = ForecasterRecursive(
                        regressor = LGBMRegressor(random_state=123),
                        lags      = 6
                    )
        return forecaster
    
    def create_svr_regresor_forecaster(self, df, y_column, steps):
        forecaster = ForecasterRecursive(
                    regressor = SVR(kernel='rbf', C=0.5, epsilon=0.1),
                    lags      = 6
                )
        return forecaster

    def create_sarimax_forecaster(self, df, y_column, steps):
        forecaster = ForecasterRecursive(
                    regressor = ForecasterSarimax(kernel='rbf', C=0.5, epsilon=0.1),
                    lags      = 6
                )
        return forecaster
    
    def hyperparameter_grid_earch(self, forecaster, df, y_column, steps):
        cv = TimeSeriesFold(
                steps              = 36, 
                initial_train_size = int(len(df) * 0.5),
                fixed_train_size   = False,
                refit              = False,
        )

        param_grid = {'alpha': np.logspace(-5, 5, 10)}

        lags_grid = [5, 12, 20]

        results_grid = grid_search_forecaster(
                            forecaster         = forecaster,
                            y                  = df[y_column],
                            cv                 = cv,
                            param_grid         = param_grid,
                            lags_grid          = lags_grid,
                            metric             = 'mean_squared_error',
                            return_best        = True,
                            n_jobs             = 'auto',
                            verbose            = False
                        )
        return forecaster

class FitPredict:

    def fit_forecaster(self, forecaster, df, y_column):
        forecaster.fit(y=df[y_column])
        return forecaster
    
    def get_predictions(self, forecaster, steps):
        return forecaster.predict(steps=steps)

    
class Evaluation:

    def plot_predictions_versus_test_data(self, train, test, y_column, predictions):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        train[y_column].plot(ax=ax, label='train')
        test[y_column].plot(ax=ax, label='test')
        predictions.plot(ax=ax, label='predictions')
        ax.legend()

    def test_error(self, df, y_column, predictions):
        error_rmse = root_mean_squared_error(
                        y_true = df[y_column],
                        y_pred = predictions
                    )
        error_mape = mean_absolute_percentage_error(
                        y_true = df[y_column],
                        y_pred = predictions
                    )
        return error_rmse, error_mape
    
