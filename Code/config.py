from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from morfist import MixedRandomForest
# ------------------------ Model ------------------------
# Root directory
Lasso = {'sensor_point5_i_value': {'alpha': 1.0},
         'sensor_point6_i_value': {'alpha': 5.0},
         'sensor_point7_i_value': {'alpha': 1.0},
         'sensor_point8_i_value': {'alpha': 5.0},
         'sensor_point9_i_value': {'alpha': 7.0},
         'sensor_point10_i_value': {'alpha': 7.0}}

Ridge = {'sensor_point5_i_value': {'alpha': 500.0},
         'sensor_point6_i_value': {'alpha': 500.0},
         'sensor_point7_i_value': {'alpha': 100.0},
         'sensor_point8_i_value': {'alpha': 500.0},
         'sensor_point9_i_value': {'alpha': 500.0},
         'sensor_point10_i_value': {'alpha': 500.0}}


XGB = {'sensor_point5_i_value': {'learning_rate': 0.2,
      'max_depth': 5,
      'min_child_weight': 10,
      'n_estimators': 30},
     'sensor_point6_i_value': {'learning_rate': 0.2,
      'max_depth': 20,
      'min_child_weight': 10,
      'n_estimators': 30},
     'sensor_point7_i_value': {'learning_rate': 0.2,
      'max_depth': 13,
      'min_child_weight': 10,
      'n_estimators': 30},
     'sensor_point8_i_value': {'learning_rate': 0.2,
      'max_depth': 20,
      'min_child_weight': 3,
      'n_estimators': 500},
     'sensor_point9_i_value': {'learning_rate': 0.2,
      'max_depth': 20,
      'min_child_weight': 3,
      'n_estimators': 30},
     'sensor_point10_i_value': {'learning_rate': 0.2,
      'max_depth': 5,
      'min_child_weight': 10,
      'n_estimators': 30}}

KN = {'sensor_point5_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'uniform'},
     'sensor_point6_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'distance'},
     'sensor_point7_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'distance'},
     'sensor_point8_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'uniform'},
     'sensor_point9_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'distance'},
     'sensor_point10_i_value': {'leaf_size': 20,
      'n_neighbors': 7,
      'weights': 'distance'}}


Ada = {'sensor_point5_i_value': {'learning_rate': 0.2, 'n_estimators': 500},
        'sensor_point6_i_value': {'learning_rate': 0.2, 'n_estimators': 300},
        'sensor_point7_i_value': {'learning_rate': 0.5, 'n_estimators': 500},
        'sensor_point8_i_value': {'learning_rate': 0.2, 'n_estimators': 500},
        'sensor_point9_i_value': {'learning_rate': 0.5, 'n_estimators': 500},
        'sensor_point10_i_value': {'learning_rate': 0.5, 'n_estimators': 500}}

Y_variable = ['sensor_point5_i_value']

pipeline = ([('poly',PolynomialFeatures(degree = 2)),
             ('std_scaler', StandardScaler()),
             ('Model', xgb.XGBRegressor(**XGB[Y_variable]))
            ])

order_list = [5, 3, 2, 1, 4, 0]


# -

# ## Convert notebook to python script

# convert notebook.ipynb to a .py file
# !jupytext --to py config.ipynb