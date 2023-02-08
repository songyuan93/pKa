from sklearn.ensemble import RandomForestRegressor
genome =   {'n_estimators':100, 'criterion':"MAE",
            'max_depth': 1, 'min_samples_split': 2, 
            'min_samples_leaf':1, 'min_weight_fraction_leaf': 0.0, 
            'max_features': None, 'max_leaf_nodes': 1, 
            'min_impurity_decrease': 0.0, 'bootstrap':True, 
            'n_jobs': 1, 'random_state': 1 }

#vincent part
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR


from sklearn.ensemble import AdaBoostRegressor


# Logan's part

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
genomeLDA = {'solver':"svd", 'shrinkage':None, 'priors':None, 'n_components':None, 
             'store_covariance':False, 'tol':0.0001, 'covariance_estimator':None}

from sklearn.ensemble import ExtraTreesRegressor
genomeETR = {'n_estimators':100, 'criterion':"squared_error", 'max_depth':None, 
             'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 
             'max_features':1.0, 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 
             'bootstrap':False, 'oob_score':False, 'n_jobs':None, 'random_state':None, 
             'verbose':0, 'warm_start':False, 'ccp_alpha':0.0, 'max_samples':None}

from sklearn.neighbors import KNeighborsRegressor
genomeKNN = {'n_neighbors':5, 'weights':"uniform", 'algorithm':"auto", 
             'leaf_size':30, 'p':2, 'metric':"minkowski", 'metric_params':None, 'n_jobs':None}

from sklearn.linear_model import RidgeCV, SGDRegressor
genomeRCV = {'alphas':(0.1, 1.0, 10.0), 'fit_intercept':True, 'scoring':None, 'cv':None, 
             'gcv_mode':None, 'store_cv_values':False, 'alpha_per_target':False}

genomeSGDR = {'loss':"squared_error", 'penalty':"l2", 'alpha':0.0001, 'l1_ratio':0.15, 
              'fit_intercept':True, 'max_iter':1000, 'tol':0.001, 'shuffle':True, 'verbose':0, 
              'epsilon':0.1, 'random_state':None, 'learning_rate':"invscaling", 'eta0':0.01, 
              'power_t':0.25, 'early_stopping':False, 'validation_fraction':0.1, 'n_iter_no_change':5, 
              'warm_start':False, 'average':False}