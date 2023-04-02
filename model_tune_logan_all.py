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
from sklearn.model_selection import KFold, ShuffleSplit
# cv1 = KFold(n_samples, k=10)
# cv2 = ShuffleSplit(n_samples, test_size=.2, n_iter=20)

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



sgdParamGrid = {
    'loss':['huber', 'epsilon_insensitive'], # 'squared_error'
    'penalty':['l2', 'l1', 'elasticnet', None], 
    'alpha':[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9],
    'l1_ratio':[0.05, 0.15, 0.3, 0.5, 0.7, 0.9],
    'fit_intercept':[True], #defualt
    'max_iter':[100, 500, 1000, 5000],
    'tol':list(np.logspace(-5, -1, num=5, base=10)), 
    'shuffle':[True], #gives better rmse
    'verbose':[0, 1, 2, 3], 
    'epsilon':[0.001, 0.01, 0.1, 1, 10], 
    'random_state':[42],
    'learning_rate':['invscaling', 'adaptive'], #'optimal', 'constant',
    'eta0':[0.001, 0.01, 0.1, 1, 10], 
    'power_t':[0.0025, 0.025, 0.25, 2.5, 25], 
    'early_stopping':[False], #gives better rmse
    'validation_fraction':[0.1, 0.15, 0.2], 
    'n_iter_no_change':[5, 10, 15, 20],
    'warm_start':[False], #default
    'average':[True] #gives better rmse
}

adaParamGrid = {
    'estimator':[None],
    'n_estimators':[10, 50, 100, 200, 500, 1000], 
    'learning_rate':[0.01, 0.05, 0.1, 0.15, 0.2], 
    'loss':['linear', 'square'], # 'exponential'
    'random_state':[42],
}

knnParamGrid = {
    'n_neighbors':[1, 3, 5, 10, 15, 20], 
    'weights':['distance'],#['uniform', 'distance', None], 
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'leaf_size':[2, 5, 10, 30, 50, 100, 200], 
    'p':[1, 2], 
    'metric':['minkowski'], 
    'metric_params':[None], 
    'n_jobs':[None, 0, 3, 5, 7, 10]
}

ridgeParamGrid = {
    'alphas':[0.1, 1.0, 10.0], 
    'fit_intercept': [True, False],
    'scoring':[None],
    'cv': [None, 1, 2, 3, 4, 5], 
    'gcv_mode': [None, 'auto', 'svd', 'eigen'],
    'store_cv_values': [True], # False not important
    'alpha_per_target': [True], # False not important
}

nnParamGrid = {
    'hidden_layer_sizes':[(128,64,32)], 
    'activation':['identity'],#['relu', 'identity', 'logistic', 'tanh'], 
    'solver':['lbfgs'],#['adam', 'lbfgs', 'sgd'],
    'alpha':[0.0001],#[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9], 
    'batch_size':['auto'], 
    'learning_rate':['constant'],#['constant', 'invscaling', 'adaptive'],# default
    'learning_rate_init':[0.001],#[0.0001, 0.001, 0.1, 1, 10],
    'power_t':[0.5],#[0.005, 0.05, 0.5, 5, 50], 
    'max_iter':[5000],#[100, 200, 500, 1000, 5000], 
    'shuffle':[True],#[True, False],# default
    'random_state':[42], 
    'tol':[1e-4],#list(np.logspace(-5, -1, num=5, base=10)), 
    'verbose':[True],#[True, False],
    'warm_start':[True],#[True, False],
    'momentum':[0.9],#[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9], 
    'nesterovs_momentum':[True],#[True, False],# default
    'early_stopping':[True],#[True, False], 
    'validation_fraction':[0.1],#[0.0001, 0.1, 0.3, 0.5, 0.7, 0.9],
    'beta_1':[0.9],#[0.3, 0.5, 0.7, 0.9, 0.999], 
    'beta_2':[0.999],#[0.3, 0.5, 0.7, 0.9, 0.999], 
    'epsilon':[1e-8],#[1e-08, 0.001, 0.01, 0.1, 1, 10],
    'n_iter_no_change':[10],#[5, 10, 15, 20], 
    'max_fun':[15000],#[1000, 5000, 15000, 30000]
}



CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def saveParameterMetrics(grid_search, name):
    #save the parameter and metrics to csv file
    filename = "all_tests_logan/all_tests_"+name+".csv"
    if not os.path.exists(filename):
        f = open(filename, "a+")
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
            f.write("\n")
        f.write("\n")
    else:
        f = open(filename, "a")
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            f.write("%0.10f,%r" % (np.sqrt(-mean_score), params))
            f.write("\n")
        f.write("\n")
    f.close()

# save features_importance to csv file
def saveFeaturesImportance(grid_search, name, data):
    feature_importance = grid_search.best_estimator_#.feature_importances_
    # params_importance = grid_search.best_params_#.feature_importances_
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_attributes = list(data.select_dtypes(include=numerics))
    # num_features = len(feature_importance)

    # save the best parameters to a file with the model name
    new_folder_name = "gridsearchcv_logan"
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    # Create a new text file inside the folder
    filename = os.path.join(new_folder_name, name+".csv")
    # if not os.path.exists("gridsearchcv_logan"+name+".csv"):
    #     open(filename, 'x').close() # TODO it says file already exists
    with open(filename, 'a') as f:
        # f.write(feature_importance + sorted(num_attributes)), reverse=True)
        f.write((str(feature_importance) + " " + str(sorted(num_attributes))) + "\n")
    f.close()

def testEveryParam(model, param_grid, name, x_test, y_test):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_test, y_test)
    # bestParam = grid_search.best_params_
    # bestScore = grid_search.best_score_
    # cv_scores = grid_search.cv_results_
    saveParameterMetrics(grid_search, name)
    saveFeaturesImportance(grid_search, name, x_test)
    # fine_tune_model(model, param_grid, x_train, y_train, x_test, y_test)
    return grid_search.best_estimator_

import sys
def printTopParams(num, name):
    topParams = []
    for x in range(0, num):
        topParams.append([float(sys.maxsize), ""])

    with open("all_tests_logan/all_tests_"+name+".csv") as file:
        for line in file:
            if line == "\n" or line == "":
                continue
            line = line.split(",", 1)
            if line[0] != "nan":
                line[0] = eval(line[0])
                for x in range(0, num):
                    topParams[x][0] = float(topParams[x][0])
                    if line[0] < topParams[x][0]:
                        topParams.insert(x, line)
                        topParams.pop()
                        break
    
    for x in range(0, num):
        print(str(x+1) + '. rmse: ' + str(topParams[x][0]) + ' and params: ' + topParams[x][1])

# # creates a file if filename does not exist and appends the model name and rmse to the file
# def write_to_file(filename, model_name, metrics):
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write(model_name + "\n" + metrics + "\n")
#     else:
#         with open(filename, 'a') as f:
#             f.write(model_name + "\n" + metrics + "\n")
#     f.close()

estimators = {
    'sgd': SGDRegressor(),#make_pipeline(StandardScaler(), SGDRegressor()),
    'ada': AdaBoostRegressor(),
    'knn': KNeighborsRegressor(),
    'ridge':RidgeCV(), # classical linear regressor
    'nn': MLPRegressor(),

    # #logan's model
    # 'sgd':make_pipeline(StandardScaler(), 
    #     SGDRegressor(max_iter=1000, tol=1e-3)), # classical linear regressor
    # 'ada':AdaBoostRegressor(random_state=209, n_estimators=30,estimator=RandomForestRegressor(n_estimators=30,random_state=209)), # ensemble based method
    # 'knn':KNeighborsRegressor(n_neighbors=2), # nearest neighbors regressor
    # 'ridge':RidgeCV(), # classical linear regressor
    # 'nn':MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=500), # neural networks model
    }

estimatorsTuning = {
    'sgd': [SGDRegressor(), sgdParamGrid],#[make_pipeline(StandardScaler(), SGDRegressor()), sgdParamGrid],
    'ada': [AdaBoostRegressor(), adaParamGrid],
    'knn': [KNeighborsRegressor(), knnParamGrid],
    'ridge': [RidgeCV(), ridgeParamGrid],
    'nn': [MLPRegressor(), nnParamGrid],
}
##########################################################################################################

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
    'n_estimators': (100, 1000), 
    'learning_rate': (0.01, 0.4), 
}

knnBounds = {
    'n_neighbors': (1, 20),  
    'leaf_size': (2, 300),
}

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
        average=True,
        early_stopping=False,
        fit_intercept=False,
        learning_rate='invscaling',
        loss='huber',
        penalty=None,
        shuffle=True,
        warm_start=False,

    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

def ada_evaluate(n_estimators, learning_rate):
    model = AdaBoostRegressor(
        n_estimators=int(n_estimators), 
        learning_rate=(learning_rate),
        estimator=None,
        loss='square',
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

def knn_evaluate(n_neighbors, leaf_size):
    model = KNeighborsRegressor(
        n_neighbors=int(n_neighbors),  
        leaf_size=int(leaf_size),
        algorithm='auto',
        metric='minkowski',
        metric_params=None,
        weights='distance',
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

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
        activation='identity',
        batch_size='auto',
        early_stopping=True,
        learning_rate='constant',
        nesterovs_momentum=True,
        shuffle=True,
        solver='lbfgs',
        verbose=True,
        warm_start=True,
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse

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
    filename = os.path.join(new_folder_name, "all_tests_"+name+".csv")
    
    if not os.path.exists("bayesian_optimization_logan/all_tests_"+name+".csv"):
        open(filename, 'x').close()

    with open(filename, 'a') as f:
        f.write(str(-optimizer.max['target'])+","+str(optimizer.max['params']) + "\n")
    f.close()

    # same code to save in different file
    new_folder_name = "all_tests_logan"
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    filename = os.path.join(new_folder_name, name+".csv")
    if not os.path.exists("all_tests_logan/"+"all_tests_"+name+".csv"):
        open(filename, 'x').close()
    with open(filename, 'a') as f:
        f.write(str(-optimizer.max['target'])+","+str(optimizer.max['params']) + "\n")
    f.close()
    return optimizer.max

estimatorsBayesian = {

        #logan's model
        # 'sgd': [sgd_evaluate, sgdBounds],
        'ada': [ada_evaluate, adaBounds],
        'knn': [knn_evaluate, knnBounds],
        'nn': [nn_evaluate, nnBounds],
        # 'gp':GPR(kernel=kernel,n_restarts_optimizer=9).fit(x_train,y_train)
}

##########################################################################################################
print("Enter either number to begin:")
print("1 to run one or all models using gridsearchcv")
print("2 to run one or all models using bayesian optimization")
print("3 to find the best parameters for one or all models")
user = int(input(">>> "))

if user == 1:
    print("Enter one of the following names to run:")
    print("sgd")
    print("ada")
    print("knn")
    print("ridge")
    print("nn")
    print("all")
    user = input(">>> ")

    if user == 'sgd' or user == 'ada' or user == 'knn' or user == 'ridge' or user == 'nn':
        model = estimatorsTuning[user][0]
        params = estimatorsTuning[user][1] 
        print(user)
        testEveryParam(model, params, user, x_test, y_test) # tests and writes every combination

    elif user == 'all':
        for name, estimator in estimators.items():
            model = estimatorsTuning[name][0]
            params = estimatorsTuning[name][1] 
            print(name)
            testEveryParam(model, params, name, x_test, y_test) # tests and writes every combination

if user == 2:
    print("Enter one of the following names to run:")
    print("sgd")
    print("ada")
    print("knn")
    print("nn")
    print("all")
    user = input(">>> ")

    if user == 'sgd' or user == 'ada' or user == 'knn' or user == 'nn':
        fittedModel = estimatorsBayesian[user][0]
        bounds = estimatorsBayesian[user][1]
        print(user)
        bayesianOptimization(bounds, fittedModel, user)
        # bestTargetParams = bayesianOptimization(bounds, fittedModel, user)
        # model = estimatorsTuning[user][0]
        # checkImprovement(user, model, bestTargetParams)

    elif user == 'all':
        for name, estimator in estimatorsBayesian.items():
            fittedModel = estimator[0]
            bounds = estimator[1]
            print(name)
            bayesianOptimization(bounds, fittedModel, name)
            # bestTargetParams = bayesianOptimization(bounds, fittedModel, name)
            # model = estimatorsTuning[name][0]
            # checkImprovement(name, model, bestTargetParams)

elif user == 3:
    print("Enter one of the following names to find best parameters:")
    print("sgd")
    print("ada")
    print("knn")
    print("ridge")
    print("nn")
    print("all")
    user = input(">>> ")

    if user == 'sgd' or user == 'ada' or user == 'knn' or user == 'ridge' or user == 'nn':
        print("Enter the number of top params you would like for " + user + ":")
        num = int(input(">>> "))
        printTopParams(num, user) # finds and prints out the number of specified top params in order

    elif user == 'all':
        print("Enter the number of top params you would like for each model:")
        num = int(input(">>> "))
        for name, estimator in estimators.items():
            print(name)
            printTopParams(num, name) # finds and prints out the number of specified top params in order