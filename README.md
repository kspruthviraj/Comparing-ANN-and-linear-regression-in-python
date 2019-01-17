# Compare outputs of single neuron (Gradient-descent) and simple linear regression
## Using Python script 

A script that runs gradient descent algorithm on single neuron and compares RMSE with normal linear regression model


This script saves the correlation matrix between different features as jpg file and the plot between single neuron prediction and linear regression prediction

# how to run
Place the NN02_PK.py and *.csv file in the same folder or change the path of the file in the script

**python NN02_PK.py**


**NOTE:Here the script with only two important features RM and LSTAT, as found by data exploration. 
Other features can be included but uncommenting the line 39 in the script and commenting the line 38**



# Explanation of gradient descent

If we have a simple linear equation such as y= w1x1 +w2x2 as shown in below figure 
![y](https://github.com/kspruthviraj/Gradient-descent/blob/master/y.png) 

We have to find w1 and w2 when x1, x2 and y are given such that the above equation is true.
If we simply guess the value of w1 and w2, then we would get y but might not be equal to the original y. Let's call this as y_hat
![y-hat](https://github.com/kspruthviraj/Gradient-descent/blob/master/y_hat.png) 

We can calulate how much the new y_hat, that was found by guessing w1 and w2, differs from the original y.
To do this we introduce cost, where cost=0.5*(y-y_hat)^2
![cost](https://github.com/kspruthviraj/Gradient-descent/blob/master/cost.png) 

Now, our objective is to minimize this cost. By simply differentiating cost with respect to w1 and w2 we get gradients.
![Gradients](https://github.com/kspruthviraj/Gradient-descent/blob/master/Gradients.png) 

Now to find w1 and w2 where the cost reaches to minimum, we start taking small steps towards the direction of minima.
This is nothing but gradient descent
![Gradient_descent](https://github.com/kspruthviraj/Gradient-descent/blob/master/Gradient_descent.png) 

This is looped over different epochs.


# Sample output from ipython

#### Build single neuron with Gradient descent algorithm #########
epoch :0 cost:45170.756384
epoch :5000 cost:6116.193370
epoch :10000 cost:6116.193370
Single neuron RMSE on train data is 5.502561676233308
Single neuron RMSE on test data is 5.6642524724292835

################### Linear regression ################################
Linear regression RMSE on train data is 5.5025616762333085
Linear regression RMSE on test data is 5.664252472426871


The RMSE of the single neuron with Gradient descent algorithm is quite similar to Linear regression.


