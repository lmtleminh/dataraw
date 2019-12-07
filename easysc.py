# -*- coding: utf-8 -*-
"""
This is an easy binning solution for credit scorecard build.

Author : Tri Le <lmtleminh@gmail.com>

Categorical binning for Credit Scoring

This is an easy categorical binning solution for credit scorecard build. It is designed to
group the optimal categories by utilizing the 
which is only applied on factor variables.

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib import parallel_backend
import sys

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
        
    def _superbin(self, X, y):
        #set seed
        np.random.seed(self.random_state)
        #bootstrap sampling
        X_t = np.concatenate((np.random.choice(X.flatten(), 
                                               (X.shape[0], self.n_iter
                                                ), replace = True), X), axis = 1)
        
        #run tree in parallel
        with parallel_backend('threading', n_jobs = self.n_jobs):
            thresholds = Parallel()(delayed(self._treeBin)(x.reshape(-1, 1), y) for x in X_t.T)
        
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
        
            if cnt[i] / len(X) < self.p / 100: 
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
                if abs(bad_rate_1[j] - bad_rate_1[j-1]) < self.min_rate/100: #self
                    thres_1 = np.delete(thres_1, j-1 if j == len(thres_1) else j)
                    j = 0
                else:
                    j += 1
            else:
                j += 1
            
        return thres_1
                       
    def _treeBin(self, X, y):  
        clf = DecisionTreeClassifier(max_depth = 2, random_state = self.random_state)
        clf.fit(X, y)
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

    def fit(self, X, y):
        if not isinstance(X, (pd.core.frame.DataFrame, 
                              pd.core.series.Series, np.ndarray)):
            raise ValueError('Invalid data object')
        
        y = check_array(y)
        self.thres_ = {}
        if isinstance(X, pd.core.frame.DataFrame):
            n = 0
            for i in X:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, n+1, X.shape[1]))
                sys.stdout.flush()
                n += 1
                if X[i].dtype == 'object':
                    continue
                else:
                    try:
                        X_t = X[i].values.reshape(-1,1)#VarianceThreshold(self.threshold).fit_transform(X[i].values.reshape(-1,1))
                        self.thres_[i] = self._superbin(X_t, y)
                    except:
                        sys.stdout.write(' No feature in X meets the variance threshold 0.00000\n')
                        continue
                
        if isinstance(X, pd.core.series.Series):
            sys.stdout.write('Processing : %s, %s out of %s.\n' % (X.name, 1, 1))
            sys.stdout.flush()
            if X.dtype != 'object':
                X_t = VarianceThreshold(self.threshold).fit_transform(X.values.reshape(-1,1))
                self.thres_[X.name] = self._superbin(X_t, y)
        if isinstance(X, np.ndarray):
            try:
                X.shape[1] == 1
            except:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                raise ValueError('Expected 2D array, got 1D array instead:\
                                 Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.')
            if X.shape[1] == 1:
                sys.stdout.write('Processing : %s, %s out of %s.\n' % (0, 1, 1))
                sys.stdout.flush()
                X_t = VarianceThreshold(self.threshold).fit_transform(X)
                self.thres_[0] = self._superbin(X_t, y)
            else:
                for i in range(len(X.T)):
                    sys.stdout.write('Processing : %s, %s out of %s.\n' % (i, i+1, len(X.T)))
                    sys.stdout.flush()
                    self.thres_[i] = self._superbin(X[:,i].reshape(-1,1), y)
        sys.stdout.write('Done! \n')
        return self
        
            
            
#        ohe = OneHotEncoder()
#        ohe.fit(X.reshape(-1,1))
#        X_t = ohe.transform(X.reshape(-1,1)).toarray()
#        clf = DecisionTreeClassifier()
#        clf.fit(X_t, y)
#        tree.plot_tree(clf)
#        
##testthat
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
#
#dd = pd.read_csv('D:/Internal Score/iFTSv4.3/data_raw.csv')
#df = dd.drop(['MIS_DATE', 'APP_ID'], axis = 1)
#df = df.assign(BAD = lambda x: np.where(x.MAX_DPD >= 90, 1, np.where(x.MAX_DPD <= 10, 0, -1)))
#df.drop('MAX_DPD', axis = 1, inplace = True)
#df = df.loc[df.BAD != -1]
#
#X = df.BIRTH_DAYS.values.reshape(-1, 1)
#y = df.BAD.values.reshape(-1, 1)
#
#np.random.choice(X.flatten(), (X.shape[0], 10))
#
#X[np.random.choice(X.shape[0], X.shape[0], replace = False),:].shape
#
#scBin = ScBinning()

