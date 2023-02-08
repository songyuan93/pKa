from sklearn.ensemble import RandomForestRegressor
genome =   {'n_estimators':100, 'criterion':"MAE",
            'max_depth': 1, 'min_samples_split': 2, 
            'min_samples_leaf':1, 'min_weight_fraction_leaf': 0.0, 
            'max_features': None, 'max_leaf_nodes': 1, 
            'min_impurity_decrease': 0.0, 'bootstrap':True, 
            'n_jobs': 1, 'random_state': 1 }

#vincent part

#######################################################
##build neural network model
#import the machine learning model package
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

def MLPR(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None, genome=None):

    '''
    Fit a Multi-layer Perceptron Regressor (MLPRegressor) model to the training data and return the model.
    
    Parameters:
    @parm x_train: training dataset
    @parm y_true: label of training dataset 
    @parm x_test: testing dataset
    @parm y_test: label of testing dataset
    @parm pred: bool, give prediction of test set or not
    @parm plot: bool, whether plot reference vs. prediction scatter plot
    @parm save_name: the name of the saved figure 
    @param genome: contains the parameters 

    Returns:
    model (MLPRegressor): Trained MLPRegressor model
    '''
    

    #build up model
    model = MLPRegressor(**genome).fit(x_train,y_train)

    #predict 
    if pred:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)

    #plot
    if plot:
        plot_group(y_train,y_test,
                   y_pred_test=y_pred_test,
                   y_pred_train=y_pred_train,
                   save_name=save_name)
    
    return model

keys_to_keep = ['random_state']
genomeMLPR = {k: v for k, v in genome.items() if k in keys_to_keep}
#training the model to test the model correction
mlpr = MLPR(x_train=x_train,y_train=y_train,x_test=x_test,
         y_test=y_test,pred=True,plot=True, genome=genomeMLPR)

#######################################################

#######################################################
from sklearn.ensemble import GradientBoostingRegressor

def GBR(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None, genome=None):
    
    """
    Fit a Gradient Boosting Regressor model to the training data and return the model.
    
    Parameters:
    @parm x_train: training dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_true: label of training dataset 
    @parm x_test: testing dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_test: label of testing dataset
    @parm pred: bool, give prediction of test set or not
    @parm plot: bool, whether plot reference vs. prediction scatter plot
    @parm save_name: the name of the saved figure 
    @param genome: contains the parameters below
    @parm n_estimators (int, optional): Number of trees in the forest. Default is 100.
    @parm learning_rate (float, optional): Learning rate shrinks the contribution of each tree by learning_rate. 
                                     There is a trade-off between learning_rate and n_estimators.
    @parm max_depth (int, optional): Maximum depth of the individual regression estimators. The maximum
                               depth limits the number of nodes in the tree. Tune this parameter
                               for best performance; the best value depends on the interaction
                               of the input variables.
    @parm random_state (int, optional): Seed for random number generation. Default is 0.

    Returns:
    model (GradientBoostingRegressor): Trained gradient boosting regressor model
    """

    model = GradientBoostingRegressor(**genome).fit(x_train, y_train)

    #predict
    if pred:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)

    #plot
    if plot:
        plot_group(y_train,y_test,
                   y_pred_test=y_pred_test,
                   y_pred_train=y_pred_train,
                   save_name=save_name)

    return model

keys_to_keep = ['n_estimators']
genomeGBR = {k: v for k, v in genome.items() if k in keys_to_keep}
gbr = GBR(x_train=x_train,y_train=y_train,x_test=x_test,
         y_test=y_test,pred=True,plot=True, genome=genomeGBR)

#######################################################

#######################################################

from sklearn.svm import SVR

def SVR(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None, genome=None):

    """
    Fit a Support Vector Regression (SVR) model to the training data and return the model.
    
    Parameters:
    @parm x_train: training dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_true: label of training dataset 
    @parm x_test: testing dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_test: label of testing dataset
    @parm pred: bool, give prediction of test set or not
    @parm plot: bool, whether plot reference vs. prediction scatter plot
    @parm save_name: the name of the saved figure 
    @param genome: contains the parameters below
    @parm kernel (str, optional): Specifies the kernel type to be used in the algorithm. 
                            It must be one of 'linear', 'poly', 'rbf' or 'sigmoid'. Default is 'linear'.
    @parm C (float, optional): Penalty parameter C of the error term. It controls the trade off between 
                        smooth decision boundary and classifying the training points correctly.
    @parm epsilon (float, optional): Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within 
                               which no penalty is associated in the training loss function with points
                               predicted within a distance epsilon from the actual value.
    @parm degree (int, optional): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    @parm gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 
                                    If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) 
                                    as value of gamma. If 'auto', uses 1 / n_features.

    Returns:
    model (SVR): Trained support vector regression model
    """

    #build up model and fit
    model = SVR(**genome).fit(x_train,y_train)

    #predict
    if pred:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
    #plot
    if plot:
        plot_group(y_train,y_test,
                   y_pred_test=y_pred_test,
                   y_pred_train=y_pred_train,
                   save_name=save_name)

    return model

keys_to_keep = ['random_state']
genomeSVR = {k: v for k, v in genome.items() if k in keys_to_keep}
#training the model to test the model correction
svr = SVR(x_train=x_train,y_train=y_train,x_test=x_test,
         y_test=y_test,pred=True,plot=True, genome=genomeSVR)

#######################################################

#######################################################

from sklearn.ensemble import AdaBoostRegressor

def adaboostRegressor(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None, genome=None):
    """
    Fit an AdaBoost Regressor model to the training data and return the model.
    
    Parameters:
    @parm x_train: training dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_true: label of training dataset 
    @parm x_test: testing dataset, Feature matrix of shape (n_samples, n_features)
    @parm y_test: label of testing dataset
    @parm pred: bool, give prediction of test set or not
    @parm plot: bool, whether plot reference vs. prediction scatter plot
    @parm save_name: the name of the saved figure 
    @param genome: contains the parameters below
    @param n_estimators (int, optional): The maximum number of estimators at which boosting is terminated. 
                                  In case of perfect fit, the learning procedure is stopped early.
    @param learning_rate (float, optional): Learning rate shrinks the contribution of each regressor by learning_rate.
    @param random_state (int, optional): Seed for random number generation. Default is 0.

    Returns:
    model (AdaBoostRegressor): Trained AdaBoost regressor model
    """
    model = AdaBoostRegressor(**genome)
    model.fit(x_train, y_train)

    #predict
    if pred:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
    #plot
    if plot:
        plot_group(y_train,y_test,
                   y_pred_test=y_pred_test,
                   y_pred_train=y_pred_train,
                   save_name=save_name)

    return model

keys_to_keep = ['random_state', 'n_estimators']
genomeABR = {k: v for k, v in genome.items() if k in keys_to_keep}

abr = adaboostRegressor(x_train=x_train,y_train=y_train,x_test=x_test,
         y_test=y_test,pred=True,plot=True, genome=genomeABR)

#######################################################

#######################################################
         
# Logan's part

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import RidgeCV, SGDRegressor