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
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import ast
from catboost import CatBoostRegressor
# ------------------------ Model ------------------------

# Gridsearch超參數CV

Ridgecv = RidgeCV(alphas = [0.01,0.1,0.5,1,5,7,10,30,100,500], cv = 5)

Lassocv = LassoCV(alphas = [0.01,0.1,0.5,1,5,7,10,30,100,500], cv = 5, max_iter = 10000)

grid_search_xgbm = GridSearchCV(xgb.XGBRegressor(),
                                  param_grid = {'learning_rate':np.array([0.2]),
                                                'n_estimators':np.array([30, 100, 200, 500]),
                                                'max_depth':np.array([3, 5, 13, 20]),
                                                'min_child_weight':np.array([3, 10]),
                                                'random_state':[1]
                                                },
                                  n_jobs = -1,
                                  scoring = 'neg_root_mean_squared_error',
                                  cv = 5
                                  )
    
grid_search_KN = GridSearchCV(KNeighborsRegressor(),
                                  param_grid = {'n_neighbors':np.array([2, 3, 5, 7]),
                                                'leaf_size':np.array([20, 30, 50]),
                                                'weights':['uniform','distance']
                                               },
                                  n_jobs = -1,
                                  scoring = 'neg_root_mean_squared_error',
                                  cv = 5
                                  )

grid_search_Ada = GridSearchCV(AdaBoostRegressor(),
                               param_grid = {'n_estimators':np.array([50, 300, 500]),
                                             'learning_rate':np.array([0.2, 0.5]),
                                             'random_state':[1]
                                             },
                               n_jobs = -1,
                               scoring = 'neg_root_mean_squared_error',
                               cv = 5
                              )

grid_search_Cat = GridSearchCV(CatBoostRegressor(),
                               param_grid = {'iterations': [100, 200],
                                             'learning_rate': [0.03, 0.1],
                                             'depth': [2, 5, 8],
                                             'l2_leaf_reg': [0.5, 3],
                                             'eval_metric':['RMSE'],
                                             'random_state':[1]
                                            },
                               n_jobs = -1,
                               scoring = 'neg_root_mean_squared_error',
                               cv = 5
                              )


# 模型超參數
Lasso_params = {'sensor_point5_i_value': {'alpha': 1.0}, 
                'sensor_point6_i_value': {'alpha': 5.0}, 
                'sensor_point7_i_value': {'alpha': 1.0}, 
                'sensor_point8_i_value': {'alpha': 5.0},
                'sensor_point9_i_value': {'alpha': 7.0}, 
                'sensor_point10_i_value': {'alpha': 7.0}}

Ridge_params = {'sensor_point5_i_value': {'alpha': 500.0}, 
                'sensor_point6_i_value': {'alpha': 500.0}, 
                'sensor_point7_i_value': {'alpha': 100.0}, 
                'sensor_point8_i_value': {'alpha': 500.0}, 
                'sensor_point9_i_value': {'alpha': 500.0}, 
                'sensor_point10_i_value': {'alpha': 500.0}}

XGB_params = {'sensor_point5_i_value': {'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point6_i_value': {'learning_rate': 0.2, 'max_depth': 20, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point7_i_value': {'learning_rate': 0.2, 'max_depth': 13, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point8_i_value': {'learning_rate': 0.2, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 500, 'random_state': 1}, 
              'sensor_point9_i_value': {'learning_rate': 0.2, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point10_i_value': {'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}}

KN_params = {'sensor_point5_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'uniform'}, 
             'sensor_point6_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point7_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point8_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'uniform'}, 
             'sensor_point9_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point10_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}}

Ada_params = {'sensor_point5_i_value': {'learning_rate': 0.2, 'n_estimators': 50, 'random_state': 1}, 
              'sensor_point6_i_value': {'learning_rate': 0.5, 'n_estimators': 50, 'random_state': 1}, 
              'sensor_point7_i_value': {'learning_rate': 0.5, 'n_estimators': 50, 'random_state': 1}, 
              'sensor_point8_i_value': {'learning_rate': 0.2, 'n_estimators': 50, 'random_state': 1}, 
              'sensor_point9_i_value': {'learning_rate': 0.5, 'n_estimators': 50, 'random_state': 1}, 
              'sensor_point10_i_value': {'learning_rate': 0.5, 'n_estimators': 50, 'random_state': 1}}

Cat_params = {'sensor_point5_i_value': {'depth': 5, 'eval_metric': 'RMSE', 'iterations': 100, 'l2_leaf_reg': 0.5, 'learning_rate': 0.1, 'random_state': 1}, 
              'sensor_point6_i_value': {'depth': 5, 'eval_metric': 'RMSE', 'iterations': 200, 'l2_leaf_reg': 0.5, 'learning_rate': 0.1, 'random_state': 1}, 
              'sensor_point7_i_value': {'depth': 2, 'eval_metric': 'RMSE', 'iterations': 200, 'l2_leaf_reg': 0.5, 'learning_rate': 0.1, 'random_state': 1}, 
              'sensor_point8_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 200, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point9_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 100, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point10_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 200, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}}



# 依序跑模型最佳順序
order_list = [5, 3, 2, 1, 4, 0]

# 使用的模型list
model_name = ['Ridge','Lasso','XGB','KN','Ada','Cat']

# Y變數選取
Y_variable = 'sensor_point6_i_value'

# 放入超參數後的model
model_info = {'Ridge':{'CV':{Ridgecv},
                       'Model': Ridge(**ast.literal_eval(f'{Ridge_params}')[f'{Y_variable}'])
                      },
              
              'Lasso':{'CV':{Lassocv},
                       'Model': Lasso(**ast.literal_eval(f'{Lasso_params}')[f'{Y_variable}'])
                      },

              'XGB':{'CV':{grid_search_xgbm},
                     'Model': xgb.XGBRegressor(**ast.literal_eval(f'{XGB_params}')[f'{Y_variable}'])
                    },
              
              'KN':{'CV':{grid_search_KN},
                    'Model': KNeighborsRegressor(**ast.literal_eval(f'{KN_params}')[f'{Y_variable}'])
                   },
              
              'Ada':{'CV':{grid_search_Ada},
                     'Model': AdaBoostRegressor(**ast.literal_eval(f'{Ada_params}')[f'{Y_variable}'])
                    },
              
              'Cat':{'CV':{grid_search_Cat},
                     'Model': CatBoostRegressor(**ast.literal_eval(f'{Cat_params}')[f'{Y_variable}'])
                    },
             }


# -

# ## Convert notebook to python script

# convert notebook.ipynb to a .py file
# !jupytext --to py config.ipynb
