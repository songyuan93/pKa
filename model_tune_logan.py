# sckit learn package
import matplotlib.pyplot as plt #plotting package
import numpy as np  #data manipulate package
import pandas as pd #data loading package
#import the metrics test machine learning models
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold


# from xgboost import XGBRegressor
# import lightgbm as ltb

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF,ConstantKernel, WhiteKernel

from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import os


def get_dataset(filename):
    df = pd.read_csv(filename)
    #one hot encoder for residue
    df = pd.get_dummies(df,columns=['ResName'])
    #get testing data
    X = df.iloc[:,3:].copy()
    X = X.loc[:,X.columns!="pKa"]
    #get testing label
    y = df.loc[:,'pKa'].copy()
    
    return X, y

# test = pd.read_csv("dataset/testset_WMa_predpka_byXGBWMa.csv")
# print(test.head())

# pka = test.loc[:,'pKa']
# pred_pka = test.loc[:,'pred_pKa']

wt_mt_asn_train = "dataset/training_set_WT+MT+aSN.txt"
wt_mt_asn_test = "dataset/test_set_WT+MT+aSN.txt"

#training the model to test the model correction
x_train, y_train = get_dataset(wt_mt_asn_train)
x_test, y_test = get_dataset(wt_mt_asn_test)

# create an extensive parameter grid for gradientboostingregressor
# """ gboostParamGrid = {
#     'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
#     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
#     'max_depth': [3, 5, 7, 9, 11, 13],
#     'min_samples_split': [2, 3, 4, 5, 6],
#     'min_samples_leaf': [1, 2, 3, 4],
#     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     'loss': ['ls', 'lad', 'huber', 'quantile'],
#     'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
#     'init': [None, 'zero', 'mean', 'median'],
#     'verbose': [0, 1, 2, 3],
#     'validation_fraction': [0.1, 0.15, 0.2],
#     'n_iter_no_change': [5, 10, 15, 20],
#     'tol': [1e-4, 1e-5, 1e-6]
# } """

# gboostParamGrid = {
#     'n_estimators': [50, 100, 200, 300, 500],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.5, 0.7, 0.9, 1.0],
#     'min_samples_split': [2, 5, 10, 20, 50],
#     'min_samples_leaf': [1, 2, 4, 8, 16],
#     'max_depth': [3, 5, 7, 9],
#     'alpha': list(np.logspace(-5, 3, num=9, base=10)),
#     'loss': ['ls', 'lad', 'huber', 'quantile'],
#     'criterion': ['friedman_mse', 'mse', 'mae'],
#     'random_state': [42]
# }

# xgboostParamGrid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
#     'max_depth': [3, 5, 7, 9],
#     'min_child_weight': [1, 3, 5, 7],
#     'gamma': [0, 0.1, 0.2, 0.3, 0.4],
#     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'reg_alpha': [0, 0.1, 0.5, 1, 2],
#     'reg_lambda': [0, 0.1, 0.5, 1, 2],
#     'n_estimators': [30, 50, 100, 150, 200, 250, 300],
#     'objective': ['reg:squarederror', 'reg:linear', 'reg:gamma'],
#     'booster': ['gbtree', 'gblinear', 'dart'],
#     'tree_method': ['auto', 'exact', 'approx', 'hist'],
#     'max_delta_step': [0, 1, 2, 3],
#     'scale_pos_weight': [1, 2, 5, 10],
#     'base_score': [0.25, 0.5, 0.75],
#     'random_state': [42]
# }

# lgbmParamGrid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
#     'num_leaves': [5, 10, 15, 20, 25, 31, 50, 75, 100, 150],
#     'max_depth': [-1, 3, 5, 7, 9, 11],
#     'min_child_samples': [10, 20, 30, 50, 100],
#     'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'reg_alpha': [0, 0.1, 0.5, 1, 2],
#     'reg_lambda': [0, 0.1, 0.5, 1, 2],
#     'n_estimators': [50, 100, 150, 200, 250, 300],
#     'objective': ['regression', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error'],
#     'boosting_type': ['gbdt', 'dart', 'goss'],
#     'max_bin': [63, 127, 255, 511, 1023],
#     'min_data_in_leaf': [5, 10, 15, 20, 25, 31, 50, 75, 100],
#     'extra_trees': [False, True],
#     'path_smooth': [0, 1, 10, 100],
#     'random_state': [42]
# }

# svrParamGrid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#     'gamma': ['scale', 'auto'] + list(np.logspace(-5, 2, num=8, base=10)),
#     'epsilon': [0.001, 0.01, 0.1, 1, 10],
#     'shrinking': [True, False],
#     'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'coef0': list(np.logspace(-3, 3, num=7, base=10)),
#     'max_iter': [100, 500, 1000, 5000],
#     'tol': list(np.logspace(-5, -1, num=5, base=10)),
#     'cache_size': [100, 500, 1000, 2000, 5000],
#     'decision_function_shape': ['ovr', 'ovo']
# }

# etrParamGrid = {
#     'n_estimators': [10, 50, 100, 200, 500, 1000],
#     'criterion': ['mse', 'mae'],
#     'max_depth': [None] + list(range(5, 110, 5)),
#     'min_samples_split': [2, 5, 10, 20, 50, 100, 200],
#     'min_samples_leaf': [1, 2, 4, 8, 16],
#     'min_impurity_decrease': list(np.logspace(-5, -1, num=5, base=10)),
#     'splitter': ['best', 'random'],
#     'bootstrap': [True, False],
#     'oob_score': [True, False],
#     'warm_start': [True, False],
#     'random_state': [42]
# }

# logan's ParamGrids
sgdParamGrid = {
    'loss':['squared_error', 'huber', 'epsilon_insensitive'], 
    'penalty':['l2', 'l1', 'elasticnet', None], 
    'alpha':[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9],
    'l1_ratio':[0.05, 0.15, 0.3, 0.5, 0.7, 0.9],
    'fit_intercept':[True, False], 
    'max_iter':[100, 500, 1000, 5000],
    'tol':list(np.logspace(-5, -1, num=5, base=10)), 
    'shuffle':[True, False], 
    'verbose':[0, 1, 2, 3], 
    'epsilon':[0.001, 0.01, 0.1, 1, 10], 
    'random_state':[42],
    'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0':[0.001, 0.01, 0.1, 1, 10], 
    'power_t':[0.0025, 0.025, 0.25, 2.5, 25], 
    'early_stopping':[True, False], 
    'validation_fraction':[0.1, 0.15, 0.2], 
    'n_iter_no_change':[5, 10, 15, 20],
    'warm_start':[True, False],
    'average':[True, False]
}

adaParamGrid = {
    'estimator':[None],
    'n_estimators':[10, 50, 100, 200, 500, 1000], 
    'learning_rate':[0.01, 0.05, 0.1, 0.15, 0.2], 
    'loss':['linear', 'square', 'exponential'], 
    'random_state':[42],
}

knnParamGrid = {
    'n_neighbors':[1, 3, 5, 10, 15, 20], 
    'weights':['uniform', 'distance', None], 
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'leaf_size':[2, 5, 10, 30, 50, 100, 200], 
    'p':[1, 2], 
    'metric':['minkowski'], 
    'metric_params':[None], 
    'n_jobs':[None, 0, 3, 5, 7, 10]
}

# ridgeParamGrid = {
#     # 'alphas':[(0.1, 1.0, 10.0)], 
#     'fit_intercept': [True, False],
#     'scoring':[None],
#     'cv': [None, 1, 2, 3, 4, 5], 
#     'gcv_mode': [None, 'auto', 'svd', 'eigen'],
#     'store_cv_values': [True, False], 
#     'alpha_per_target': [True, False],
# }

nnParamGrid = {
    # 'hidden_layer_sizes':(100,), 
    'activation':['relu', 'identity', 'logistic', 'tanh'], 
    'solver':['adam', 'lbfgs', 'sgd'], 
    'alpha':[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9], 
    'batch_size':['auto'], 
    'learning_rate':['constant', 'invscaling', 'adaptive'], 
    'learning_rate_init':[0.0001, 0.001, 0.1, 1, 10],
    'power_t':[0.005, 0.05, 0.5, 5, 50], 
    'max_iter':[100, 200, 500, 1000, 5000], 
    'shuffle':[True, False], 
    'random_state':[42], 
    'tol':list(np.logspace(-5, -1, num=5, base=10)), 
    'verbose':[True, False],
    'warm_start':[True, False],
    'momentum':[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9], 
    'nesterovs_momentum':[True, False], 
    'early_stopping':[True, False], 
    'validation_fraction':[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9],
    'beta_1':[0.3, 0.5, 0.7, 0.9, 0.999], 
    'beta_2':[0.3, 0.5, 0.7, 0.9, 0.999], 
    'epsilon':[1e-08, 0.001, 0.01, 0.1, 1, 10],
    'n_iter_no_change':[5, 10, 15, 20], 
    'max_fun':[1000, 5000, 15000, 30000]
}


# gboostBounds = {
#     'n_estimators': (10, 1000),
#     'learning_rate': (0.01, 1),
#     'max_depth': (1, 10),
#     'min_samples_split': (2, 50),
#     'min_samples_leaf': (1, 20),
#     'subsample': (0.1, 1.0),
#     'alpha': (0.01, 0.99),
# }

# xgboostBounds = {
#     'n_estimators': (1, 1000),  # number of boosting stages to perform
#     'learning_rate': (0.01, 1),  # learning rate shrinks the contribution of each tree
#     'max_depth': (1, 20),  # maximum depth of a tree
#     'min_child_weight': (1, 20),  # minimum sum of instance weight (hessian) needed in a child
#     'subsample': (0.1, 1),  # fraction of samples to be used for fitting the individual base learners
#     'colsample_bytree': (0.1, 1),  # fraction of columns to be randomly subsampled for each tree
#     'gamma': (0, 5),    # minimum loss reduction required to make a further partition on a leaf node
#     'reg_alpha': (0, 1),    # L1 regularization term on weights
#     'reg_lambda': (0, 1)    # L2 regularization term on weights
# }

# lgbmBounds = {
#     'learning_rate': (0.001, 0.1),
#     'max_depth': (2, 15),
#     'num_leaves': (10, 300),
#     'feature_fraction': (0.1, 1),
#     'bagging_fraction': (0.1, 1),
#     'bagging_freq': (1, 10),
#     'min_child_samples': (5, 50),
#     'lambda_l1': (0, 1),
#     'lambda_l2': (0, 1)
# }

# svrBounds = {
#     'C': (0.1, 100),
#     'gamma': (0.0001, 10),
#     'epsilon': (0.01, 1),
#     'degree': (1, 10),
# }

# etrBounds = {
#     'n_estimators': (10, 1000),
#     'max_features': (0.1, 1),
#     'max_depth': (2, 50),
#     'min_samples_split': (2, 50),
#     'min_samples_leaf': (1, 10)
# }

#TODO create ParamBounds (used for Bayesian optimization)
sgdBounds = { 
    'alpha': (0.000001, 0.99),
    'l1_ratio': (0.05, 0.99), 
    'max_iter': (100, 5000), 
    'tol': (0.000001, 0.1),
    'verbose': (0, 3),
    'epsilon': (0.000001, 50),  
    'eta0': (0.000001, 10),
    'power_t': (0.0025, 25),
    'validation_fraction': (0.00001, 0.2),
    'n_iter_no_change': (5, 30)
}

adaBounds = { 
    'n_estimators': (10, 1000), 
    'learning_rate': (0.01, 0.2), 
}

knnBounds = {
    'n_neighbors': (1, 20),  
    'leaf_size': (2, 300),
}

# ridgeBounds = {     
# }

nnBounds = { # ConvergencWarning: Stochastic Optimizer: Maximum iterations (100)" (, same with (133)) "and the optimization hasn't converged yet."
    'alpha': (0.0001, 0.9),   
    'learning_rate_init': (0.0001, 10), 
    'power_t': (0.005, 50), 
    'max_iter': (100, 5000), 
    'tol': (0.00001, 0.1),
    'momentum': (0.0001, 0.9), 
    'validation_fraction': (0.0001, 0.9),
    'beta_1': (0.3, 0.999), 
    'beta_2': (0.3, 0.999), 
    'epsilon': (1e-08, 10),
    'n_iter_no_change': (5, 20),
    'max_fun': (1000, 30000),
}

# kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(10)  + WhiteKernel(5)
estimators = {
        #Vincent's model
        # 'gboost': GradientBoostingRegressor(),
        # 'xgboost':XGBRegressor(n_estimators=30),
        # 'lgbm':ltb.LGBMRegressor(n_estimators=30),
        # 'svr':make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        # 'etr':ExtraTreesRegressor(n_estimators=30,random_state=209),
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),

        #logan's model
        'sgd':make_pipeline(StandardScaler(), 
            SGDRegressor(max_iter=1000, tol=1e-3)), # classical linear regressor
        'ada':AdaBoostRegressor(random_state=209, n_estimators=30,estimator=RandomForestRegressor(n_estimators=30,random_state=209)), # ensemble based method
        'knn':KNeighborsRegressor(n_neighbors=2), # nearest neighbors regressor
        'ridge':RidgeCV(), # classical linear regressor
        'nn':MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=500), # neural networks model
        # 'gp':GPR(kernel=kernel,n_restarts_optimizer=9).fit(x_train,y_train)
        }

estimatorsTuning = {
        #Vincent's model
        # 'gboost': [GradientBoostingRegressor(), gboostParamGrid],
        # 'xgboost':[XGBRegressor(), xgboostParamGrid],
        # 'lgbm':[ltb.LGBMRegressor(), lgbmParamGrid],
        # 'svr':[make_pipeline(StandardScaler(), SVR()), svrParamGrid],
        # 'etr':[ExtraTreesRegressor(), etrParamGrid],
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),

        #logan's model
        'sgd': [make_pipeline(StandardScaler(), SGDRegressor()), sgdParamGrid],
        'ada': [AdaBoostRegressor(), adaParamGrid],
        'knn': [KNeighborsRegressor(), knnParamGrid],
        'nn': [MLPRegressor(), nnParamGrid],
        # 'ridge': [RidgeCV(), ridgeParamGrid]
}

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def saveParamterMetrics(grid_search, name):
    #save the parameter and metrics to csv file
    filename = "hyperparameter_fine_tuning_metrics_logan/"+name+".csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
                f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
                f.write("\n")
    with open(filename, 'a') as f:
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
            f.write("\n")
    f.close()

# save features_importance to csv file
def saveFeaturesImportance(grid_search, name, data):
    feature_importance = grid_search.best_estimator_.feature_importances_
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_attributes = list(data.select_dtypes(include=numerics))
    # num_features = len(feature_importance)
    filename = "features_importance_logan/"+name+".csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(sorted(zip(feature_importance, num_attributes), reverse=True))
            
    with open(filename, 'a') as f:
        f.write(sorted(zip(feature_importance, num_attributes), reverse=True))
    f.close()
        

def hyperparamterTuning(model, param_grid, name, x_test, y_test):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_test, y_test)
    bestParam = grid_search.best_params_
    bestScore = grid_search.best_score_
    cv_scores = grid_search.cv_results_
    saveParamterMetrics(grid_search, name)
    saveFeaturesImportance(grid_search, name, x_test)
    # fine_tune_model(model, param_grid, x_train, y_train, x_test, y_test)
    return grid_search.best_estimator_

# cross validate the model and return the rmse mean
def cross_validate(model, train, test):
    scores = cross_val_score(model, train, test, cv=10, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    # format a return string that includes rmse, mae, and r2 to 10 decimal places with commas
    return "{:.10f}".format(rmse_scores.mean())

# creates a file if filename does not exist and appends the model name and rmse to the file
def write_to_file(filename, model_name, metrics):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(model_name + "\n" + metrics + "\n")
    else:
        with open(filename, 'a') as f:
            f.write(model_name + "\n" + metrics + "\n")
    f.close()

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    # print('Model Performance')
    # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    # print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def checkImprovement(name, base_model, optimal):
    base_accuracy = np.sqrt(mean_squared_error(base_model.fit(x_train,y_train).predict(x_test), y_test))
    improved_accuracy = -optimal["target"]
    filename = "improvements_logan.txt"
    
    if not os.path.exists(filename):
        open(filename, 'x').close()

    with open(filename, 'a') as f:
        f.write(name + "\nImprovement of {:0.4f}%.\n".format( 100 * (improved_accuracy - base_accuracy) / base_accuracy))
        f.write("with parameters: " + str(optimal["params"]) + "\n\n")
    f.close()
    # print('Improvement of {:0.2f}%.'.format( 100 * (improved_accuracy - base_accuracy) / base_accuracy))

from bayes_opt import BayesianOptimization
# fine tune model with bayesian optimization
def bayesianOptimization(bounds, fittedModel, name):
    optimizer = BayesianOptimization(
        f=fittedModel,
        pbounds=bounds,
        random_state=24,
    )
    optimizer.maximize(init_points=10, n_iter=500)
    # save the best parameters to a file with the model name
    new_folder_name = "bayesian_optimization_logan"
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

    # Create a new text file inside the folder
    filename = os.path.join(new_folder_name, name+".csv")
    
    if not os.path.exists("bayesian_optimization_logan/"+name+".csv"):
        open(filename, 'x').close()

    with open(filename, 'a') as f:
        f.write(str(-optimizer.max['target'])+","+str(optimizer.max['params']) + "\n")
    f.close()
    return optimizer.max
    # optimizer.max

# def gb_regression_cv(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, alpha):
#     model = GradientBoostingRegressor(
#         n_estimators=int(n_estimators),
#         learning_rate=learning_rate,
#         max_depth=int(max_depth),
#         min_samples_split=int(min_samples_split),
#         min_samples_leaf=int(min_samples_leaf),
#         subsample=subsample,
#         alpha=alpha,
#         random_state=56
#     )
#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

#     return rmse

# def xgb_evaluate(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
#     # create the XGBRegressor model with the specified hyperparameters
#     model = XGBRegressor(
#         objective='reg:squarederror',
#         n_estimators=int(n_estimators),
#         learning_rate=learning_rate,
#         max_depth=int(max_depth),
#         min_child_weight=int(min_child_weight),
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         gamma=gamma,
#         reg_alpha=reg_alpha,
#         reg_lambda=reg_lambda,
#         n_jobs=-1
#     )

#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

#     return rmse

# def lgbm_evaluate(learning_rate, max_depth, num_leaves, feature_fraction, bagging_fraction, bagging_freq, min_child_samples, lambda_l1, lambda_l2):
#     # create the model with the specified hyperparameters
#     model = ltb.LGBMRegressor(
#         learning_rate=learning_rate,
#         max_depth=int(max_depth),
#         num_leaves=int(num_leaves),
#         feature_fraction=max(min(feature_fraction, 1), 0),
#         bagging_fraction=max(min(bagging_fraction, 1), 0),
#         bagging_freq=int(bagging_freq),
#         min_child_samples=int(min_child_samples),
#         lambda_l1=lambda_l1,
#         lambda_l2=lambda_l2
#     )
    
#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

#     return rmse

# def svr_evaluate(C, gamma, epsilon, degree):
#     # create the pipeline with a StandardScaler and SVR model
#     model = make_pipeline(StandardScaler(), SVR(C=C, gamma=gamma, epsilon=epsilon, degree=int(degree)))
    
#     # Perform 10-fold cross-validation
#     kf = KFold(n_splits=10, shuffle=True, random_state=56)
#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', error_score='raise').mean()

#     return rmse

# def extratree_evaluate(n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf):
#     # create the ExtraTreesRegressor model with the specified hyperparameters
#     model = ExtraTreesRegressor(
#         n_estimators=int(n_estimators),
#         max_features=max_features,
#         max_depth=int(max_depth),
#         min_samples_split=int(min_samples_split),
#         min_samples_leaf=int(min_samples_leaf),
#         random_state=56
#     )
#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

#     return rmse

#TODO create method: sgd_evaluate
def sgd_evaluate(alpha, l1_ratio, max_iter, tol, verbose, epsilon, eta0, power_t, validation_fraction, n_iter_no_change):
    model = SGDRegressor(
        alpha=(alpha),
        l1_ratio=(l1_ratio),
        max_iter=int(max_iter),
        tol=(tol),
        verbose=int(verbose),
        epsilon=(epsilon),
        eta0=(eta0),
        power_t=(power_t),
        validation_fraction=(validation_fraction),
        n_iter_no_change=int(n_iter_no_change),
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

#TODO create method: ada_evaluate
def ada_evaluate(n_estimators, learning_rate):
    model = AdaBoostRegressor(
        n_estimators=int(n_estimators), 
        learning_rate=(learning_rate),
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

#TODO create method: knn_evaluate
def knn_evaluate(n_neighbors, leaf_size):
    model = KNeighborsRegressor(
        n_neighbors=int(n_neighbors),  
        leaf_size=int(leaf_size),
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

#TODO create method: ridge_evaluate
# def ridge_evaluate():
#     model = RidgeCV(
#     )
#     # use cross-validation to estimate the model's RMSE
#     rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

#     return rmse

#TODO create method: nn_evaluate
def nn_evaluate(alpha, learning_rate_init, power_t, max_iter, tol, momentum, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun):
    model = MLPRegressor(
        alpha=(alpha),   
        learning_rate_init=(learning_rate_init), 
        power_t=(power_t), 
        max_iter=int(max_iter), 
        tol=(tol),
        momentum=(momentum),
        validation_fraction=(validation_fraction),
        beta_1=(beta_1),
        beta_2=(beta_2),
        epsilon=(epsilon),
        n_iter_no_change=int(n_iter_no_change),
        max_fun=int(max_fun),
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

""" 
# gradient boosting regressor
def tune_gboost():
    best_estimator = hyperparamterTuning(GradientBoostingRegressor(),gboostParamGrid,"gboost", x_train, y_train)
    checkImprovement("gboost", GradientBoostingRegressor().fit(x_train,y_train), best_estimator, x_test, y_test)
    y_pred_test = best_estimator.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_pred_test,y_test)) #get root mean square error
    mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
    r2 = r2_score(y_pred_test,y_test) #get r square value
    metrics = "RMSE: {}\n".format(rmse) + "MAE: {}\n".format(mae) + "r2: {}\n".format(r2)
    write_to_file("test_results.txt", "gboost", metrics)
    # write_to_file("test_results.txt", "gboost", rmse)

# tune_gboost()

def tuneAll():
    for name, estimator in estimatorsTuning.items():
        # perform a grid search on each model
        # save parameters and score
        # with the best model predict on test set
        # print out the results
        best_estimator = hyperparamterTuning(estimator[0], name, estimator[1], x_train, y_train)
        y_pred_test = best_estimator.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_pred_test,y_test)) #get root mean square error
        mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
        r2 = r2_score(y_pred_test,y_test) #get r square value

        metrics = "RMSE: {}\n".format(rmse) + "MAE: {}\n".format(mae) + "r2: {}\n".format(r2)
        write_to_file("test_results.txt", name, metrics)

def runInitial():
    for name, estimator in estimators.items():
        
        print(name)
        model = estimator.fit(x_train,y_train)
        y_pred_test = model.predict(x_test)
        rmse = mean_squared_error(y_pred_test,y_test)**0.5 #get root mean square error
        mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
        r2 = r2_score(y_pred_test,y_test) #get r square value

        print("RMSE: ",rmse)
        print("MAE: ", mae)
        print("r2: ",r2)
        print("")
        # format a return string metrics that includes rmse, mae, and r2 seperated by new lines
        metrics = "RMSE: {}\n".format(rmse) + "MAE: {}\n".format(mae) + "r2: {}\n".format(r2)
        write_to_file("test_results.txt", name, metrics)

# runInitial() """

# save model
import pickle
def save_model(model, model_name):
    filename = model_name + '.bin'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        file.close

estimatorsBayesian = {
        #Vincent's model
        # 'gboost': [gb_regression_cv, gboostBounds],
        # 'xgboost':[xgb_evaluate, xgboostBounds],
        # 'lgbm':[lgbm_evaluate, lgbmBounds],
        # 'svr':[svr_evaluate, svrBounds],
        # 'etr':[extratree_evaluate, etrBounds],
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),

        #logan's model
        'sgd': [sgd_evaluate, sgdBounds],
        'ada': [ada_evaluate, adaBounds],
        'knn': [knn_evaluate, knnBounds],
        'nn': [nn_evaluate, nnBounds],
        # 'ridge': [ridge_evaluate, ridgeBounds],
        # 'gp':GPR(kernel=kernel,n_restarts_optimizer=9).fit(x_train,y_train)
}

for name, estimator in estimatorsBayesian.items():
    fittedModel = estimator[0]
    bounds = estimator[1]
    print(name)
    bestTargetParams = bayesianOptimization(bounds, fittedModel, name)
    model = estimatorsTuning[name][0]
    checkImprovement(name, model, bestTargetParams)
    # model = fittedModel(**best_params)
    # model.fit(x_train, y_train, **best_params)
    # feature_importances = model.feature_importances_
    # np.savetxt(name+'_feature_importances.csv', feature_importances, delimiter=',')
    # save_model(model, name)
