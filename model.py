'''
Author:Songyuan Yao
date: 01/31/2023
comments: models for ensemble learning base on scikit learn
'''

# sckit learn package
import matplotlib.pyplot as plt #plotting package
import numpy as np  #data manipulate package
import pandas as pd #data loading package
#import the metrics test machine learning models
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

#load the training set
train_filename = "dataset/training_set_WT.txt" #ensure a correct address
train = pd.read_csv(train_filename)
x_train = train.iloc[:,5:-2].copy()
y_train = train.iloc[:,-1].copy()

#load the test data set
test_filename = "dataset/test_set_WT.txt" #ensure a correct address
test = pd.read_csv(test_filename)
x_test = test.iloc[:,5:-2].copy()
y_test = test.iloc[:,-1].copy()

def plot_scatter(f1,f2,title,bound=20):

    '''
    plot scatter figure reference vs prediction
    @parm f1: true label
    @parm f2: prediction label
    @parm title: title for plot 
    @parm bound: the bound for plot

    '''
    rmse = mean_squared_error(f1,f2)**0.5 #get root mean square error
    mae = mean_absolute_error(f1,f2) #get mean absolute error
    r2 = r2_score(f1,f2) #get r square value
    error = np.array(f1 - f2)

    # fig,ax = plt.subplots(1,2)
    ax = plt.gca()
    
    sc = ax.scatter(f1,f2,marker='o',c=error,cmap='Spectral',
                    edgecolors='k',linewidths=1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_aspect('equal', adjustable='box')

    ax.plot([0, bound], [0, bound], color="firebrick", linewidth=1.2)
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_title(title,size =18) 

    ax.text(0.2,11.5,"RMSE: {}".format("{:.3f}".format(rmse)),size = 13)
    ax.text(0.2,11,"MAE: {}".format("{:.3f}".format(mae)),size = 13)
    ax.text(0.2,10.5,r"$r^2$: {}".format("{:.3f}".format(r2)),size = 13)
    ax.set_xlabel("Reference pKa",size =15)
    ax.set_yticks(np.arange(0, 13, 2))
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    
    ax.set_ylabel("Predicted pKa",size =15)
    plt.colorbar(sc,fraction=0.046, pad=0.04)


def plot_group(y_train,y_test,y_pred_train,y_pred_test,save_name):
    '''
    A figure panel with training and testing data prediction
    The example is avaible in overleaf project figure folder file named nn.png
    @parm y_train: true label for training set
    @parm y_test: true label for test set
    @parm y_pred_train: training set prediciton
    @parm y_pred_test: test set prediction
    @parm save_name: the figure name you want to save like "nn.png"
    '''
    fig,ax = plt.subplots(1,2,figsize=(13,6))
    plt.subplot(121)
    plot_scatter(f1=y_train,f2=y_pred_train,title="Training set",bound=12)
    plt.subplot(122)
    plot_scatter(f1=y_test,f2=y_pred_test,title="Test set",bound=12)
    plt.tight_layout()
    plt.savefig(save_name,dpi=300)  

#Neural network example
#######################################################
##build neural network model
#import the machine learning model package
from sklearn.neural_network import MLPRegressor

def NN(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):
    '''
    return linear regression model
    @parm x_train: training dataset
    @parm y_true: label of training dataset 
    @parm x_test: testing dataset
    @parm y_test: label of testing dataset
    @parm pred: bool, give prediction of test set or not
    @parm plot: bool, whether plot reference vs. prediction scatter plot
    @parm save_name: the name of the saved figure 
    '''
    
    #build up model
    model = MLPRegressor().fit(x_train,y_train)

    #predict 
    if pred:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        # rmse = RMSE(y_test,y_pred)

    #plot
    if plot:
        plot_group(y_train,y_test,
                   y_pred_test=y_pred_test,
                   y_pred_train=y_pred_train,
                   save_name=save_name)
    
    return model

#training the model to test the model correction
nn = NN(x_train=x_train,y_train=y_train,x_test=x_test,
         y_test=y_test,pred=True,plot=True)

#######################################################

#######################################################
#SVM
#import package

#writing the function
def SVM(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################

#######################################################
# Linear Discriminant Analysis (LDA)
#import package

#writing the function
def LDA(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None

#training the model to test the model correction

#######################################################


#######################################################
#random forest
#import package

#writing the function
def RF(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################


#######################################################
#Bagging Tree (BT)
#import package

#writing the function
def BT(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################

#Logan's part
#######################################################
#k-nearest neighbor (KNN)
#import package

#writing the function
def KNN(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################


#######################################################
#kernal ridge regression (KRR)
#import package

#writing the function
def KRR(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None

#training the model to test the model correction

#######################################################


#######################################################
#Gaussion Process (GP)
#import package

#writing the function
def GP(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################

#######################################################
#Stochastic Gradient Descent (SGD)
#import package

#writing the function
def SGD(x_train,y_train,x_test=None,
       y_test=None,pred=None,plot=None,
       save_name=None):

    #build up model

    #predict

    #plot

    return None
#training the model to test the model correction

#######################################################