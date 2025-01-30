# Data manipulation
# ==============================================================================
import numpy as np

# Modeling and Forecasting
# ==============================================================================
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterSarimax
from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import grid_search_forecaster, grid_search_sarimax
from skforecast.sarimax import Sarimax
from skforecast.exceptions import LongTrainingWarning
from skforecast.direct import ForecasterDirect
from skforecast.recursive._forecaster_recursive_multiseries import ForecasterRecursiveMultiSeries

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

    def create_random_forest_regresor_forecaster(self, steps):
        forecaster = ForecasterRecursive(
                        regressor = RandomForestRegressor(random_state=123),
                        lags      = steps
                    )
        return forecaster
    
    def create_lgbm_regressor_forecaster(self, steps):
        forecaster = ForecasterRecursive(
                        regressor = LGBMRegressor(random_state=123),
                        lags      = steps
                    )
        return forecaster
    
    def create_svr_regresor_forecaster(self, steps):
        forecaster = ForecasterRecursive(
                    regressor = SVR(kernel='rbf', C=0.5, epsilon=0.1),
                    lags      = steps
                )
        return forecaster

    def create_sarimax_forecaster(self, steps):
        forecaster = ForecasterSarimax(
                    regressor = Sarimax(
                        order   = (1, 1, 1),
                        seasonal_order=(1, 1, 1, 12))
                        )
        return forecaster
    
    def hyperparameter_grid_search_lgbm(self, forecaster, df, y_column, steps):
        cv = TimeSeriesFold(
                steps              = steps, 
                initial_train_size = int(len(df) * 0.5),
                fixed_train_size   = False,
                refit              = False,
        )

        param_grid = {'alpha': np.logspace(-5, 5, 10)}

        lags_grid = [steps]

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
    
    def hyperparameter_grid_search_forest(self, forecaster, df, y_column, steps):
        cv = TimeSeriesFold(
                steps              = steps, 
                initial_train_size = int(len(df) * 0.5),
                fixed_train_size   = False,
                refit              = False,
        )

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }

        lags_grid = [562]

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
    
    def hyperparameter_grid_search_svr(self, forecaster, df, y_column, steps):
        cv = TimeSeriesFold(
                steps              = steps, 
                initial_train_size = int(len(df) * 0.5),
                fixed_train_size   = False,
                refit              = False,
        )

        param_grid = {
            'C': [0.1, 1, 10, 100],           # Regularization parameter
            'epsilon': [0.01, 0.1, 0.5, 1],   # Epsilon in the epsilon-SVR model
            'kernel': ['linear', 'rbf'],      # Kernel type
            'gamma': ['scale', 'auto']        # Kernel coefficient
        }

        lags_grid = [562]

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
    
    def hyperparameter_grid_search_sarimax(self, forecaster, df, y_column, steps):
        cv = TimeSeriesFold(
                steps              = steps, 
                initial_train_size = int(len(df) * 0.5),
                fixed_train_size   = False,
                refit              = True,
        )

        param_grid = {
            'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
            'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 12), (1, 1, 1, 12)],
            'trend': [None, 'n', 'c']
        }
        
        results_grid = grid_search_sarimax(
                   forecaster            = forecaster,
                   y                     = df[y_column],
                   cv                    = cv,
                   param_grid            = param_grid,
                   metric                = 'mean_absolute_error',
                   return_best           = False,
                   n_jobs                = 'auto',
                   suppress_warnings_fit = True,
                   verbose               = False,
                   show_progress         = True
        )       
        
        return forecaster

class Multiseries:

    def create_fit_multi_forecaster(self, steps, df):
        forecaster = ForecasterRecursiveMultiSeries(
                 regressor          = RandomForestRegressor(random_state=123),
                 lags               = steps,
                 transformer_series = StandardScaler(),
                 transformer_exog   = None,
                 weight_func        = None,
                 series_weights     = None
             )
        
        forecaster.fit(series=df)
        return forecaster

    def predict_multi(self, steps, forecaster, level):
        prediction_level = forecaster.predict(steps=steps, levels=level)
        return prediction_level     
        
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
    
