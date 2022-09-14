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
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import ast
from catboost import CatBoostRegressor
# ------------------------ Model ------------------------

# Gridsearch超參數CV

# Ridgecv = RidgeCV(alphas = [0.01,0.1,0.5,1,5,7,10,30,100,500], cv = 5)
# Lassocv = LassoCV(alphas = [0.01,0.1,0.5,1,5,7,10,30,100,500], cv = 5, max_iter = 10000)

grid_search_Lasso = GridSearchCV(Lasso(),
                                  param_grid = {'alpha':[0.01,0.1,0.5,1,5,7,10,30,100,500]},
                                  n_jobs = -1,
                                  scoring = 'neg_root_mean_squared_error',
                                  cv = 5
                                  )

grid_search_Ridge = GridSearchCV(Ridge(),
                                  param_grid = {'alpha':[0.01,0.1,0.5,1,5,7,10,30,100,500]},
                                  n_jobs = -1,
                                  scoring = 'neg_root_mean_squared_error',
                                  cv = 5
                                  )


grid_search_XGB = GridSearchCV(xgb.XGBRegressor(),
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
                               param_grid = {'n_estimators':np.array([100, 300, 500]),
                                             'learning_rate':np.array([0.2, 0.5]),
                                             'random_state':[1]
                                             },
                               n_jobs = -1,
                               scoring = 'neg_root_mean_squared_error',
                               cv = 5
                              )

grid_search_Cat = GridSearchCV(CatBoostRegressor(),
                               param_grid = {'iterations': [500 ,1000],
                                             'learning_rate': [0.03],
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
Lasso_params = {'sensor_point5_i_value': {'alpha': 500}, 
                'sensor_point6_i_value': {'alpha': 100}, 
                'sensor_point7_i_value': {'alpha': 500}, 
                'sensor_point8_i_value': {'alpha': 500}, 
                'sensor_point9_i_value': {'alpha': 500}, 
                'sensor_point10_i_value': {'alpha': 500}}

Ridge_params = {'sensor_point5_i_value': {'alpha': 0.1}, 
                'sensor_point6_i_value': {'alpha': 0.1}, 
                'sensor_point7_i_value': {'alpha': 0.1}, 
                'sensor_point8_i_value': {'alpha': 0.1}, 
                'sensor_point9_i_value': {'alpha': 1}, 
                'sensor_point10_i_value': {'alpha': 0.5}}

XGB_params = {'sensor_point5_i_value': {'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point6_i_value': {'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point7_i_value': {'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 500, 'random_state': 1}, 
              'sensor_point8_i_value': {'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 30, 'random_state': 1}, 
              'sensor_point9_i_value': {'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 500, 'random_state': 1}, 
              'sensor_point10_i_value': {'learning_rate': 0.2, 'max_depth': 20, 'min_child_weight': 10, 'n_estimators': 30, 'random_state': 1}}

KN_params = {'sensor_point5_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point6_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point7_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point8_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point9_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}, 
             'sensor_point10_i_value': {'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}}

Ada_params = {'sensor_point5_i_value': {'learning_rate': 0.2, 'n_estimators': 100, 'random_state': 1}, 
              'sensor_point6_i_value': {'learning_rate': 0.2, 'n_estimators': 100, 'random_state': 1}, 
              'sensor_point7_i_value': {'learning_rate': 0.5, 'n_estimators': 100, 'random_state': 1}, 
              'sensor_point8_i_value': {'learning_rate': 0.5, 'n_estimators': 300, 'random_state': 1}, 
              'sensor_point9_i_value': {'learning_rate': 0.2, 'n_estimators': 300, 'random_state': 1}, 
              'sensor_point10_i_value': {'learning_rate': 0.2, 'n_estimators': 100, 'random_state': 1}}

Cat_params = {'sensor_point5_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point6_i_value': {'depth': 2, 'eval_metric': 'RMSE', 'iterations': 1000, 'l2_leaf_reg': 0.5, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point7_i_value': {'depth': 5, 'eval_metric': 'RMSE', 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point8_i_value': {'depth': 5, 'eval_metric': 'RMSE', 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point9_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 500, 'l2_leaf_reg': 0.5, 'learning_rate': 0.03, 'random_state': 1}, 
              'sensor_point10_i_value': {'depth': 8, 'eval_metric': 'RMSE', 'iterations': 500, 'l2_leaf_reg': 0.5, 'learning_rate': 0.03, 'random_state': 1}}



# 依序跑模型最佳順序
order_list = [5, 3, 2, 1, 4, 0]

# 使用的模型list
model_name = ['Ridge','Lasso','XGB','KN','Ada','Cat']

# Y變數選取
Y_variable = 'sensor_point6_i_value'

data_y_col = ['sensor_point5_i_value','sensor_point6_i_value','sensor_point7_i_value',\
              'sensor_point8_i_value','sensor_point9_i_value','sensor_point10_i_value']


# 放入超參數後的model
model_info = {'Ridge':{'CV':{grid_search_Ridge},
                       'Model': Ridge(**ast.literal_eval(f'{Ridge_params}')[f'{Y_variable}'])
                      },
              
              'Lasso':{'CV':{grid_search_Lasso},
                       'Model': Lasso(**ast.literal_eval(f'{Lasso_params}')[f'{Y_variable}'])
                      },

              'XGB':{'CV':{grid_search_XGB},
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

# 各Y變數最佳Model(Cat/Cat/KNR/KNR/Ada/Cat)
# Cat從圖上的結果看起來還是比Lasso好，改用Cat

model_list = [Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('Cat', CatBoostRegressor(**ast.literal_eval(f'{Cat_params}')[ast.literal_eval(f'{data_y_col}')[0]]))
                        ]),
                       
              Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('Cat', CatBoostRegressor(**ast.literal_eval(f'{Cat_params}')[ast.literal_eval(f'{data_y_col}')[1]]))
                        ]),
              
              Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('KNR', KNeighborsRegressor(**ast.literal_eval(f'{KN_params}')[ast.literal_eval(f'{data_y_col}')[2]]))
                        ]),
              
              Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('KNR', KNeighborsRegressor(**ast.literal_eval(f'{KN_params}')[ast.literal_eval(f'{data_y_col}')[3]]))
                        ]),
               
              Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('Ada', AdaBoostRegressor(**ast.literal_eval(f'{Ada_params}')[ast.literal_eval(f'{data_y_col}')[4]]))
                        ]),
              
              Pipeline([('poly',PolynomialFeatures(degree = 2)),
                        ('std_scaler', StandardScaler()),
                        ('Cat', CatBoostRegressor(**ast.literal_eval(f'{Cat_params}')[ast.literal_eval(f'{data_y_col}')[5]]))
                        ])
              ]


x_train_columns = ['clean_pressure31',
 'clean_pressure41',
 'clean_pressure72',
 'clean_pressure81',
 'clean_pressure91',
 'clean_pressure92',
 'clean_pressure102',
 'oven_pa1',
 'oven_pa2',
 'oven_pb1',
 'oven_pb2',
 'oven_a3',
 'oven_b1',
 'oven_b2',
 'painting_g1_act_a_air',
 'painting_g1_act_f_air',
 'painting_g1_act_t_air',
 'painting_g1_act_hvc',
 'painting_g3_act_t_air',
 'painting_g3_act_hvv',
 'painting_g4_act_f_air',
 'painting_g4_act_hvv',
 'painting_g4_act_hvc',
 'painting_g5_act_a_air',
 'painting_g5_act_f_air',
 'painting_g6_act_a_air',
 'painting_g6_act_hvc',
 'painting_g7_act_f_air',
 'painting_g7_act_t_air',
 'painting_g7_act_hvv',
 'painting_g9_act_hvv',
 'painting_g10_act_f_air',
 'painting_g11_act_hvv',
 'painting_g12_act_a_air',
 'env_rpi05_hum',
 'env_rpi05_pm1',
 'env_rpi07_hum',
 'env_rpi07_pm10',
 'env_rpi07_temp',
 'env_rpi09_hum',
 'env_rpi09_lux',
 'env_rpi09_pm1',
 'env_rpi09_temp',
 'env_rpi14_hum',
 'env_rpi14_lux',
 'env_rpi14_pm1',
 'env_rpi14_pm10',
 'env_rpi14_temp',
 'env_rpi15_hum',
 'env_rpi15_lux',
 'env_rpi15_pm1',
 'env_rpi15_pm10',
 'env_rpi15_temp',
 'painting_g8_act_a_air_group_0',
 'painting_g8_act_a_air_group_1',
 'painting_g8_act_a_air_group_2',
 'painting_g8_act_a_air_group_3',
 'painting_g8_act_a_air_group_4',
 'painting_g9_act_hvc_group_0',
 'painting_g9_act_hvc_group_1',
 'painting_g9_act_hvc_group_2',
 'painting_g9_act_hvc_group_3',
 'painting_g9_act_hvc_group_4',
 'painting_g9_act_hvc_group_5',
 'painting_g10_act_hvv_group_0',
 'painting_g10_act_hvv_group_1',
 'painting_g10_act_hvv_group_2',
 'painting_g10_act_hvv_group_3',
 'painting_g10_act_hvv_group_4',
 'painting_g10_act_hvc_group_0',
 'painting_g10_act_hvc_group_1',
 'painting_g10_act_hvc_group_2',
 'painting_g10_act_hvc_group_3',
 'painting_g10_act_hvc_group_4',
 'painting_g10_act_hvc_group_5',
 'painting_g11_act_a_air_group_0',
 'painting_g11_act_a_air_group_1',
 'painting_g11_act_a_air_group_2',
 'painting_g11_act_a_air_group_3',
 'painting_g11_act_hvc_group_0',
 'painting_g11_act_hvc_group_1',
 'painting_g11_act_hvc_group_2',
 'painting_g11_act_hvc_group_3',
 'env_rpi05_temp_group_0',
 'env_rpi05_temp_group_1',
 'env_rpi05_temp_group_2',
 'env_rpi05_temp_group_3',
 'env_rpi05_temp_group_4',
 'env_rpi07_pm10_group_0',
 'env_rpi07_pm10_group_1',
 'env_rpi07_pm10_group_2',
 'env_rpi07_pm10_group_3',
 'env_rpi07_pm10_group_4',
 'env_rpi07_pm25_group_0',
 'env_rpi07_pm25_group_1',
 'env_rpi07_pm25_group_2',
 'env_rpi07_pm25_group_3',
 'env_rpi07_pm25_group_4',
 'env_rpi07_pm25_group_5',
 'env_rpi14_pm1_group_0',
 'env_rpi14_pm1_group_1',
 'env_rpi14_pm1_group_2',
 'env_rpi14_pm1_group_3',
 'env_rpi14_pm1_group_4',
 'env_rpi15_pm1_group_0',
 'env_rpi15_pm1_group_1',
 'env_rpi15_pm1_group_2',
 'env_rpi15_pm1_group_3',
 'env_rpi15_pm1_group_4']

# -

# ## Convert notebook to python script

# convert notebook.ipynb to a .py file
# !jupytext --to py config.ipynb
