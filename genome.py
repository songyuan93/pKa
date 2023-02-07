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

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import RidgeCV, SGDRegressor