import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def kmeans_res(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def choose_best_k_for_kmeans(scaled_data, k_range, verbose, parallel=True):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns 
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    if parallel:  
        ans = Parallel(n_jobs=-1, verbose=verbose)(delayed(kmeans_res)(scaled_data, k) for k in k_range)
        ans = list(zip(k_range, ans))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
    else:
        ans = []
        for k in k_range:
            scaled_inertia = kmeans_res(scaled_data, k)
            ans.append((k, scaled_inertia))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        
    return best_k, results