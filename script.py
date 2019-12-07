# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:18:12 2019

@author: trilm3
"""
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class ScBinning:
    """
    
    """
    def __init__(self, n_iter = 10, n_jobs = None, p = 3, min_rate = .5, 
                 threshold = 0, best = True, random_state = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.p = p
        self.min_rate = min_rate
        self.threshold = threshold
        self.best = best   
        self.random_state = random_state
        
    def _fit(#self, 
            X, y):
        #set seed
        np.random.seed(None #self.random_state
                       )
        #bootstrap sampling
        X_t = np.concatenate((np.random.choice(X.flatten(), 
                                               (X.shape[0], n_iter #self.n_iter
                                                ), replace = True), X),
                                axis = 1)
        
        #run tree in parallel
        #self.
        thresholds = Parallel(n_jobs = n_jobs #self.n_jobs
                              )(delayed(_treeBin)(x.reshape(-1, 1), y) for x in X_t.T)
        
        #calculate bad rate
        thresholds = np.hstack(thresholds)
        thresholds.sort()
        thres, cnt_thres = np.unique(thresholds, return_counts = True)
        
        i = 0
        while i <= len(thres):       
            if i == 0:
                bad_rate = np.zeros(len(thres) + 1, dtype = 'float64')
                cnt = np.zeros(len(thres) + 1, dtype = 'int64')
                bad_rate[i] = y[X <= thres[i]].mean()
                cnt[i] = len(y[X <= thres[i]])
            elif i < len(thres):
                bad_rate[i] = y[(X <= thres[i]) & (X > thres[i-1])].mean()
                cnt[i] = len(y[(X <= thres[i]) & (X > thres[i-1])])
            else:
                bad_rate[i] = y[X > thres[i-1]].mean()
                cnt[i] = len(y[X > thres[i-1]])
        
            if cnt[i] / len(X) < p: #self
                thres = np.delete(thres, i-1 if i == len(thres) else i)
                cnt_thres = np.delete(cnt_thres, i-1 if i == len(cnt_thres) else i)
                i = 0
            else:
                i += 1
        
        #trend
        ins = np.polyfit(thres, bad_rate[:-1], 1)[0]
       
        #prepare iso table
        iso_t = np.hstack((np.append(thres, thres.max() + 1).reshape(-1,1), 
                           bad_rate.reshape(-1,1)))
        iso_t = np.repeat(iso_t, np.append(cnt_thres, 1), axis = 0)
        
        #iso regression
        ir = IsotonicRegression(increasing = (ins >= 0))
        bad_rate_fit = ir.fit_transform(iso_t[:,0], iso_t[:,1])
        thresholds_t = np.hstack((iso_t[:,0].reshape(-1,1), bad_rate_fit.reshape(-1,1)))
        
        thres_1 = np.array([thresholds_t[thresholds_t[:,1] == x, 0].mean() for x in np.unique(thresholds_t[:,1])])
        thres_1.sort()
        
        j = 0
        while j <= len(thres_1):
            if j == 0:
                bad_rate_1 = np.zeros(len(thres_1) + 1, dtype = 'float64')
                bad_rate_1[j] = y[X <= thres_1[j]].mean()
            elif j < len(thres_1):
                bad_rate_1[j] = y[(X <= thres_1[j]) & (X > thres_1[j-1])].mean()
            else:
                bad_rate_1[j] = y[X > thres_1[j-1]].mean()
            
            if j > 0:
                if abs(bad_rate_1[j] - bad_rate_1[j-1]) < min_rate/100: #self
                    thres_1 = np.delete(thres_1, j-1 if j == len(thres_1) else j)
                    j = 0
                else:
                    j += 1
            else:
                j += 1
            
        return thres_1 #self
                            
    def _treeBin(#self, 
                 X, y):  
        clf = DecisionTreeClassifier(max_depth = 2, random_state = None#self.random_state
                                     )
        clf = clf.fit(X, y)
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        is_leaves = np.zeros(shape=clf.tree_.node_count, dtype=bool)
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()
            if (children_left[node_id] != children_right[node_id]):
                stack.append(children_left[node_id])
                stack.append(children_right[node_id])
            else:
                is_leaves[node_id] = True
        return threshold[~is_leaves]

    def fit(X, y):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
        
        if isinstance(X, pd.core.frame.DataFrame):
            thres_0 = np.zeros(X.shape[1])
            for i in X:
                print(i)
                if X[i].dtype == 'object':
                    continue
                
                
        
        X = VarianceThreshold(threshold).fit_transform(X)
        
        _fit(X, y)
        
class ScCatBinning:
    """
    
    """
    def __init__(self, n_iter = 10, n_jobs = None, p = 3, min_rate = .5, 
                 threshold = 0, best = True, random_state = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.p = p
        self.min_rate = min_rate
        self.threshold = threshold
        self.best = best   
        self.random_state = random_state
        
    def _impactCode(#self,
                    X, y):
        mean = y.mean()
        smooth = 300
        m = np.array([(y[X == i].sum() + smooth * mean)/(len(y[X == i]) + smooth)
                  for i in np.unique(X)])
        n = np.array([y[X == i].mean() for i in np.unique(X)])
        p = np.array([len(y[X == i])/len(X) for i in np.unique(X)])
        km_m = KMeans(n_clusters = 3)
        km_m.fit(m.reshape(-1,1))
        result = np.concatenate((km_m.labels_.reshape(-1,1), 
                                 np.unique(X).reshape(-1,1),
                                 m.reshape(-1,1),
                                 n.reshape(-1,1),
                                 p.reshape(-1,1)), axis = 1)
        result.T
        [result[result[:,0] == i] for i in np.unique(result[:,0])]
        m
        i
        np.array([len(y[X == i]) for i in np.unique(X)])[np.unique(X) == 'Vinh Phuc']
        
        nn_m = NearestNeighbors(n_neighbors = 5)
        nn_m.fit(m.reshape(-1,1))
        nn_m.kneighbors(m.reshape(-1,1), return_distance = False)
        

      
#testthat
import pandas as pd
import matplotlib.pyplot as plt
import h2o

dd = pd.read_csv('D:/Internal Score/iFTSv4.3/data_raw.csv')
df = dd.drop(['MIS_DATE', 'APP_ID'], axis = 1)
df = df.assign(BAD = lambda x: np.where(x.MAX_DPD >= 90, 1, np.where(x.MAX_DPD <= 10, 0, -1)))
df.drop('MAX_DPD', axis = 1, inplace = True)
df = df.loc[df.BAD != -1]

X = df.BIRTH_DAYS.values.reshape(-1, 1)
y = df.BAD.values.reshape(-1, 1)

np.random.choice(X.flatten(), (X.shape[0], 10))

X[np.random.choice(X.shape[0], X.shape[0], replace = False),:].shape

scBin = ScBinning()

y.shape


