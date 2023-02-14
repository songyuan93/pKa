# sckit learn package
import matplotlib.pyplot as plt #plotting package
import numpy as np  #data manipulate package
import pandas as pd #data loading package
#import the metrics test machine learning models
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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

test = pd.read_csv("dataset/testset_WMa_predpka_byXGBWMa.csv")
# print(test.head())

pka = test.loc[:,'pKa']
pred_pka = test.loc[:,'pred_pKa']

wt_mt_asn_train = "dataset/training_set_WT+MT+aSN.txt"
wt_mt_asn_test = "dataset/test_set_WT+MT+aSN.txt"

#training the model to test the model correction
x_train, y_train = get_dataset(wt_mt_asn_train)
x_test, y_test = get_dataset(wt_mt_asn_test)

# kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(10)  + WhiteKernel(5)
estimators = {
        #Vincent's model
        'gboost': GradientBoostingRegressor(),
        'xgboost':XGBRegressor(n_estimators=30),
        'lgbm':ltb.LGBMRegressor(n_estimators=30),
        'svr':make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        'etr':ExtraTreesRegressor(n_estimators=30,random_state=209),
        # 'rf':RandomForestRegressor(n_estimators=30,random_state=209),

        #logan's model
        'sgd':make_pipeline(StandardScaler(),
            SGDRegressor(max_iter=1000, tol=1e-3)),
        'ada':AdaBoostRegressor(random_state=209, n_estimators=30,base_estimator=RandomForestRegressor(n_estimators=30,random_state=209)),
        'knn':KNeighborsRegressor(n_neighbors=2),
        'ridge':RidgeCV(),
        'nn':MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=500),
        # 'gp':GPR(kernel=kernel,n_restarts_optimizer=9).fit(x_train,y_train)
        }

for name, estimator in estimators.items():
    
    print(name)
    model = estimator.fit(x_train,y_train)
    y_pred_test = model.predict(x_test)
    rmse = mean_squared_error(y_pred_test,y_test)**0.5 #get root mean square error
    mae = mean_absolute_error(y_pred_test,y_test) #get mean absolute error
    r2 = r2_score(y_pred_test,y_test) #get r square value

    print("RMSE :",rmse)
    print("MAE: ", mae)
    print("r2: ",r2)
    print("")