# sckit learn package
import matplotlib.pyplot as plt #plotting package
import numpy as np  #data manipulate package
import pandas as pd #data loading package
#import the metrics test machine learning models
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score


from xgboost import XGBRegressor
import lightgbm as ltb

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
gboostParamGrid = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    'max_depth': [3, 5, 7, 9, 11, 13],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
    'init': [None, 'zero', 'mean', 'median'],
    'verbose': [0, 1, 2, 3],
    'validation_fraction': [0.1, 0.15, 0.2],
    'n_iter_no_change': [5, 10, 15, 20],
    'tol': [1e-4, 1e-5, 1e-6]
}

xgboostParamGrid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 2],
    'n_estimators': [30, 50, 100, 150, 200, 250, 300],
    'objective': ['reg:squarederror', 'reg:linear', 'reg:gamma'],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'tree_method': ['auto', 'exact', 'approx', 'hist'],
    'max_delta_step': [0, 1, 2, 3],
    'scale_pos_weight': [1, 2, 5, 10],
    'base_score': [0.25, 0.5, 0.75],
    'random_state': [42]
}

lgbmParamGrid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'num_leaves': [5, 10, 15, 20, 25, 31, 50, 75, 100, 150],
    'max_depth': [-1, 3, 5, 7, 9, 11],
    'min_child_samples': [10, 20, 30, 50, 100],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0, 0.1, 0.5, 1, 2],
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'objective': ['regression', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error'],
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'max_bin': [63, 127, 255, 511, 1023],
    'min_data_in_leaf': [5, 10, 15, 20, 25, 31, 50, 75, 100],
    'extra_trees': [False, True],
    'path_smooth': [0, 1, 10, 100],
    'random_state': [42]
}

svrParamGrid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': ['scale', 'auto'] + list(np.logspace(-5, 2, num=8, base=10)),
    'epsilon': [0.001, 0.01, 0.1, 1, 10],
    'shrinking': [True, False],
    'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'coef0': list(np.logspace(-3, 3, num=7, base=10)),
    'max_iter': [100, 500, 1000, 5000],
    'tol': list(np.logspace(-5, -1, num=5, base=10)),
    'cache_size': [100, 500, 1000, 2000, 5000],
    'decision_function_shape': ['ovr', 'ovo']
}

etrParamGrid = {
    'n_estimators': [10, 50, 100, 200, 500, 1000],
    'criterion': ['mse', 'mae'],
    'max_depth': [None] + list(range(5, 110, 5)),
    'min_samples_split': [2, 5, 10, 20, 50, 100, 200],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['auto', 'sqrt', 'log2', None] + list(range(5, 55, 5)),
    'min_impurity_decrease': list(np.logspace(-5, -1, num=5, base=10)),
    'splitter': ['best', 'random'],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'warm_start': [True, False],
    'random_state': [42]
}


# kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(10)  + WhiteKernel(5)
estimators = {
        #Vincent's model
        'gboost': [GradientBoostingRegressor(), gboostParamGrid],
        'xgboost':[XGBRegressor(n_estimators=30), xgboostParamGrid],
        'lgbm':[ltb.LGBMRegressor(n_estimators=30), lgbmParamGrid],
        # 'svr':[make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)), svrParamGrid],
        'etr':[ExtraTreesRegressor(n_estimators=30,random_state=209), etrParamGrid],
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),

        #logan's model
        # 'sgd':make_pipeline(StandardScaler(),
        #     SGDRegressor(max_iter=1000, tol=1e-3)),
        # 'ada':AdaBoostRegressor(random_state=209, n_estimators=30,base_estimator=RandomForestRegressor(n_estimators=30,random_state=209)),
        # 'knn':KNeighborsRegressor(n_neighbors=2),
        # 'ridge':RidgeCV(),
        # 'nn':MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=500),
        # 'gp':GPR(kernel=kernel,n_restarts_optimizer=9).fit(x_train,y_train)
        }

estimatorsTuning = {
        #Vincent's model
        'gboost': [GradientBoostingRegressor(), gboostParamGrid],
        'xgboost':[XGBRegressor(), xgboostParamGrid],
        'lgbm':[ltb.LGBMRegressor(), lgbmParamGrid],
        'svr':[make_pipeline(StandardScaler(), SVR()), svrParamGrid],
        'etr':[ExtraTreesRegressor(), etrParamGrid],
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),
}

# ignore this method. not using this one
# for name, estimator in estimators.items(): fine tune the model for test dataset
def fine_tune_model(model, param_grid, x_train, y_train, x_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    y_pred_test = grid_search.predict(x_test)
    rmse = mean_squared_error(y_pred_test,y_test)**0.5 #get root mean square error
    mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
    r2 = r2_score(y_pred_test,y_test) #get r square value

    print("RMSE :",rmse)
    print("MAE: ", mae)
    print("r2: ", r2)
    print("")

def saveParamterMetrics(grid_search, name):
    #save the parameter and metrics to csv file
    filename = "hyperparameter_fine_tuning_metrics/"+name+".csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
                f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
                f.write("\n")
    with open(filename, 'a') as f:
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
            f.write("\n")

# save features_importance to csv file
def saveFeaturesImportance(grid_search, name, data):
    feature_importance = grid_search.best_estimator_.feature_importances_
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_attributes = list(data.select_dtypes(include=numerics))
    # num_features = len(feature_importance)
    filename = "features_importance/"+name+".csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(sorted(zip(feature_importance, num_attributes), reverse=True))
            
    with open(filename, 'a') as f:
        f.write(sorted(zip(feature_importance, num_attributes), reverse=True))
        

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
def write_to_file(filename, model_name, rmse):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(model_name + ", " + rmse + "\n")
    else:
        with open(filename, 'a') as f:
            f.write(model_name + ", " + rmse + "\n")


# gradient boosting regressor
def tune_gboost():
    best_estimator = hyperparamterTuning(GradientBoostingRegressor(),gboostParamGrid,"gboost", x_train, y_train)
    y_pred_test = best_estimator.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_pred_test,y_test)) #get root mean square error
    mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
    r2 = r2_score(y_pred_test,y_test) #get r square value

    write_to_file("test_results.txt", "gboost", rmse)
    
tune_gboost()

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

        write_to_file("test_results.txt", name, rmse)

# for name, estimator in estimators.items():
    
#     print(name)
#     model = estimator.fit(x_train,y_train)
#     y_pred_test = model.predict(x_test)
#     rmse = mean_squared_error(y_pred_test,y_test)**0.5 #get root mean square error
#     mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
#     r2 = r2_score(y_pred_test,y_test) #get r square value

#     print("RMSE :",rmse)
#     print("MAE: ", mae)
#     print("r2: ",r2)
#     print("")

# save model
import pickle
def save_model(model, model_name):
    filename = model_name + '.bin'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        file.close