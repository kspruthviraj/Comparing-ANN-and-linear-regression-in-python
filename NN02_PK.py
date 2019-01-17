# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:31:19 2018

@author: pruthvi
"""
""
# %% First section
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # for data exploration but not used for training artificial neuron as the exercise suggests to use only numpy
import seaborn as sns 

import sklearn
from sklearn.cross_validation import train_test_split

# %% Explore the data
print("################### first section ####################################")
print("################### Explore data ####################################")

data=pd.read_csv('mass_boston.csv') # Read the data using Pandas
data.head() # Show first 5 data points
data.isnull().sum() # Check whether there are any missing data points

correlation_matrix = data.corr().round(2) # Compute correlation matrix
sns.set(rc={'figure.figsize':(14,7)}) # Set the size of the figure
sns.heatmap(data=correlation_matrix, annot=True) # Plot correlation matrix plot
plt.savefig('Correlation.png') # Save the image

# The correlation map shows that Variables RM and LSTAT are highly correlated with MEDV
# And if MEDV is regressed against these two variables, the regression will be better.

# %% Single Neuron - Using only numpy as recommended in the exercise
print("################### second section################################")
print("#### Build single neuron with Gradient descent algorithm #########")
# Load the data and create the data matrices X and Y
X_file = np.genfromtxt('mass_boston.csv', delimiter=',', skip_header=1) 
x1 = X_file[:, (5,12)] # select only important features as found in the data exploration
#x1 = X_file[:,0:13] # select all features

y = np.array([X_file[:, 13]]).T # The target vector Y is a column of MEDV values.
N = np.shape(X_file)[0]

# Standardize the input 
x = (x1-np.mean(x1))/np.std(x1)

b=np.ones(N).reshape(N, 1); #initial b weights -
x=np.concatenate([b, x], axis=1) #  This creates a feature vector X with a column of ones (bias)

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.2, random_state = 1)

w = np.ones((x.shape[1],1))
lr = 0.001 # learning rate
n_epoch=10001 # epochs


def nn_02(x,y,w,lr,n_epoch):
    y_hat = np.dot(x,w)  # calculate y_hat for first iteration
    for epoch in range(n_epoch):  
        dcostdw = 2*0.5*np.dot(-x.T,(y-y_hat)) # derivative of cost -- gradient
        w = w - lr*dcostdw # calculate weights
        y_hat = np.dot(x,w) # calculate new y_hat using new weights after above steps
        cost = 0.5*np.sum((y - y_hat)**2)  # calculate new cost
        #cost = np.sqrt(np.mean((y_hat - y)**2))  # calculate new cost -- RMSE01
        #cost = np.sqrt((y_hat - y)**2).mean()  # calculate new cost -- RMSE02

        if epoch%5000==0:
            print("epoch :{:d} cost:{:f}".format(epoch,cost))
            y_pred=np.dot(x,w)
        if cost <= 0.01:
            print("epoch :{:d} cost:{:f}".format(epoch,cost))
            y_pred=np.dot(x,w)
            break
    return w,y_pred
  

w,y_pred_train = nn_02(X_train,Y_train,w,lr,n_epoch) # train the model
y_pred_test_neuron = np.dot(X_test,w) # calculate the predictions on the unseen dataset
#
RMSE_train = np.sqrt(np.mean((y_pred_train - Y_train)**2))  # calculate RMSE for train data
RMSE_test = np.sqrt(np.mean((y_pred_test_neuron - Y_test)**2))  # calculate RMSE for test data
print('Single neuron RMSE on train data is {}'.format(RMSE_train))
print('Single neuron RMSE on test data is {}'.format(RMSE_test))

#plt.plot(y_pred_test, Y_test, 'kx')
#plt.plot(y_pred_train, Y_train, 'kx')



# %% Linear regression
print("################### Third section ####################################")
print("################### Linear regression ################################")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train) # train the model
y_test_predict_lin_reg = lin_model.predict(X_test) # calculate the predictions on the unseen dataset

rmse_REG_train = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))  # calculate RMSE for train data
rmse_REG_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict_lin_reg)))  # calculate RMSE for test data
print('Linear regression RMSE on train data is {}'.format(rmse_REG_train))
print('Linear regression RMSE on test data is {}'.format(rmse_REG_test))


# %% Plot the predictions from two models
print("################### Fourth section ####################################")
print("################### Compare models ####################################")
np.corrcoef(y_pred_test_neuron[:,0], y_test_predict_lin_reg[:,0]) # compute the correlation coeffecicients

fig = plt.figure(figsize=(8, 4))
#plt.plot(y_pred_test_neuron, y_test_predict_lin_reg)
plt.scatter(y_pred_test_neuron, y_test_predict_lin_reg)
plt.xlabel('Predictions from single neuron')
plt.ylabel('Predictions from linear regression')
plt.title('Predictions from two models')
plt.savefig('mass_boston.png')
plt.show()



