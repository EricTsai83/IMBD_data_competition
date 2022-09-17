"""
@Author: Eric Tsai
@brief: A metric is a function that is used to judge the performance of your model. 
"""
import numpy as np



def rmse_score(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2,  axis=0))  # get mean row by row
