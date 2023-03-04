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



# ridge paramaters
ridgeParamGrid = {
    # 'alphas':[(0.1, 1.0, 10.0)], 
    'fit_intercept': [True, False],
    'scoring':[None],
    'cv': [None, 1, 2, 3, 4, 5], 
    'gcv_mode': [None, 'auto', 'svd', 'eigen'],
    'store_cv_values': [True, False], 
    'alpha_per_target': [True, False],
}

ridgeBounds = {     
}



CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def saveParamterMetrics(grid_search, name):
    #save the parameter and metrics to csv file
    filename = "gridsearchcv_logan/hyperparameter_fine_tuning_metrics_ridge.csv"
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

# from bayes_opt import BayesianOptimization
# # fine tune model with bayesian optimization
# def bayesianOptimization(bounds, fittedModel, name):
#     optimizer = BayesianOptimization(
#         f=fittedModel,
#         pbounds=bounds,
#         random_state=24,
#     )
#     optimizer.maximize(init_points=10, n_iter=500)
#     # save the best parameters to a file with the model name
#     new_folder_name = "bayesian_optimization_logan"
#     if not os.path.exists(new_folder_name):
#         os.makedirs(new_folder_name)

#     # Create a new text file inside the folder
#     filename = os.path.join(new_folder_name, name+".csv")
    
#     if not os.path.exists("bayesian_optimization_logan/"+name+".csv"):
#         open(filename, 'x').close()

#     with open(filename, 'a') as f:
#         f.write(str(-optimizer.max['target'])+","+str(optimizer.max['params']) + "\n")
#     f.close()
#     return optimizer.max
#     # optimizer.max



#TODO create method: ridge_evaluate
def ridge_evaluate():
    model = RidgeCV(
    )
    # use cross-validation to estimate the model's RMSE
    rmse = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()

    return rmse



# # gradient boosting regressor
# def tune_gboost():
#     best_estimator = hyperparamterTuning(GradientBoostingRegressor(),gboostParamGrid,"gboost", x_train, y_train)
#     checkImprovement("gboost", GradientBoostingRegressor().fit(x_train,y_train), best_estimator, x_test, y_test)
#     y_pred_test = best_estimator.predict(x_test)

#     rmse = np.sqrt(mean_squared_error(y_pred_test,y_test)) #get root mean square error
#     mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
#     r2 = r2_score(y_pred_test,y_test) #get r square value
#     metrics = "RMSE: {}\n".format(rmse) + "MAE: {}\n".format(mae) + "r2: {}\n".format(r2)
#     write_to_file("test_results.txt", "gboost", metrics)
#     # write_to_file("test_results.txt", "gboost", rmse)

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

# kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(10)  + WhiteKernel(5)
estimators = {
        'ridge':RidgeCV(), # classical linear regressor
        }

estimatorsTuning = {
        'ridge': [RidgeCV(), ridgeParamGrid]
}
# estimatorsBayesian = {
#         'ridge': [ridge_evaluate, ridgeParamGrid],
# }

for name, estimator in estimators.items():
    model = estimatorsTuning[name][0]
    params = estimatorsTuning[name][1] #ridgeParamGrid
    print(name) # ridge
    # bestTargetParams = bayesianOptimization(bounds, fittedModel, name)
    # checkImprovement(name, model, bestTargetParams)

    # TODO define this method
    # Ridge_reg = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv= 5)
    # Ridge_reg.fit(x_train, y_train)
    hyperparamterTuning(model, params, name, x_test, y_test)

    # don't use 
    # model = fittedModel(**best_params)
    # model.fit(x_train, y_train, **best_params)
    # feature_importances = model.feature_importances_
    # np.savetxt(name+'_feature_importances.csv', feature_importances, delimiter=',')
    # save_model(model, name)
