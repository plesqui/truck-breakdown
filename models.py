# This script contains the different model that we evaluated
# 1. Neural Network with one hidden layer (3 hidden units) implemented with TensorFlow.
# 2. Support Vector Machine model implemented with scikit-learn
# 3. Random Forest implemented with scikit-learn
# 4. Gaussian Process implemented with scikit-learn
# 5. Random guessing predictions


# 1. N E U R A L  N E T W O R K
# =====================================================================================

import math
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
#import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import sklearn
from sklearn import preprocessing
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF




# Create placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an input training vector (n_x = 7)
    n_y -- scalar, number of output training vector (n_y = 1)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """


    X = tf.placeholder(tf.float32, shape = [n_x,None], name = "X")
    Y = tf.placeholder(tf.float32, shape = [n_y,None], name = "Y")
   
    
    return X, Y
  
# Initialize the parameters of the network (weights and biases)

def initialize_parameters(n_input,n_hidunit,n_output):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:

                        W1 : [3, 7] (7 inputs connected to 3 hidden units)
                        b1 : [3, 1] 
                        W2 : [1, 3] (3 activations from first hidden layer connected to one output)
                        b2 : [1, 1]
                    
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2
    """
    
   
        

    W1 = tf.get_variable("W1",[n_hidunit, n_input], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_hidunit,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[n_output, n_hidunit], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [n_output,1], initializer = tf.zeros_initializer())

   

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
  
# Forward propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2",
                  the shapes are given in initialize_parameters

    Returns:
    Z2 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    
            
    Z1 = tf.add(tf.matmul(W1,X),b1)                                             
    A1 = tf.nn.relu(Z1)                                             
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                             
  
    
    return Z2
 
# Compute the cost
def compute_cost(Z2, Y):
    """
    Computes the cost
    
    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z2
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels ))
    ### END CODE HERE ###
    
    return cost
 
# Definition of the Sigmoid function
def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    

    # Create a placeholder for x. Name it 'x'.
    X = tf.placeholder(tf.float32, name = "X")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(X)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {X:z})
    

    
    return result

# Backpropagation and parameter updates
def model(X_train, Y_train, X_test, Y_test, n_hidden = 2, learning_rate = 0.001, print_cost = False, iteration = 6000):
    """
    Implements a two-layer tensorflow neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 7, number of training examples = 45)
    Y_train -- test set, of shape (output size = 1, number of training examples = 45)
    X_test -- training set, of shape (input size = 7, number of training examples = 18)
    Y_test -- test set, of shape (output size = 1, number of test examples = 18)
    learning_rate -- learning rate of the optimization
    print_cost -- True to print the cost every 100 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    
    X, Y = create_placeholders(n_x, n_y)
    

    # Initialize parameters
   
    parameters = initialize_parameters(n_x, n_hidden, n_y)
    
    
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    
    Z2 = forward_propagation(X, parameters)
   
    
    # Cost function: Add cost function to tensorflow graph
    
    cost = compute_cost(Z2, Y)
    
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for i in range(iteration):
            _ , iteration_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                

            # Print the cost every epoch
            if print_cost == True and i % 6000 == 0:
                print ("Cost after iteration %i: %f" % (i, iteration_cost))
            if print_cost == True and i % 5 == 0:
                costs.append(iteration_cost)
                


        # lets save the parameters in a variable
        parameters = sess.run(parameters)
     

        
        # Estimating the accuracy, sensitivity and RECALL
        thr = 0.5 # Threshold to classify predictions
        
        
        logits_train = np.array(Z2.eval({X: X_train}))
        sigmoid_train = sigmoid(logits_train)
        
        sigmoid_train[sigmoid_train>thr] = 1
        sigmoid_train[sigmoid_train<=thr] = 0 # This is the predicted array (training set)!!
        
        logits_test = np.array(Z2.eval({X: X_test}))
        sigmoid_test = sigmoid(logits_test)
 
        sigmoid_test[sigmoid_test>thr] = 1
        sigmoid_test[sigmoid_test<=thr] = 0 # This is the predicted array (test set)!!        
        
        
        # Testing the performance on the Test set
        tn, fp, fn, tp = confusion_matrix(Y_test.T, sigmoid_test.T).ravel()
        precission = tp/(tp+fp)
        sensitivity = tp/(tp+fn) #RECALL
        specificity = tn/(tn+fp)
  
        
        
        return parameters,sensitivity,specificity,precission

      
      
# 2. S U P P O R T   V E C T O R   M A C H I N E   M O D E L 
#===========================================================
def run_svm(X_train,Y_train,X_test,Y_test):
    classifier = svm.LinearSVC()
    classifier = svm.SVC(kernel='rbf')

    classifier.fit(X_train.T, Y_train.T)
    y_score = classifier.predict(X_test.T)
    thr = 0.5
    
    #sigmoid_test = sigmoid(np.array(y_score))
    #sigmoid_test[sigmoid_test>thr] = 1
    #sigmoid_test[sigmoid_test<=thr] = 0 # This is the predicted array!!

    tn, fp, fn, tp = confusion_matrix(Y_test.T, y_score).ravel()
    precission = tp/(tp+fp)
    sensitivity = tp/(tp+fn) #RECALL
    specificity = tn/(tn+fp)
    f1score = 2*(precission*sensitivity)/(precission+sensitivity)
    
    #print('Sensitivity on TEST set for SVM: {0:0.2f}'.format(sensitivity))
    #print('Specificity on TEST set for SVM: {0:0.2f}'.format(specificity))
    return sensitivity, specificity,precission
  
  
# 3. R A N D O M   F O R E S T   M O D E L 
# =========================================
def run_forest(X_train,Y_train,X_test,Y_test):
    classifier = RandomForestClassifier()
    classifier.fit(X_train.T,Y_train.T.ravel())    

    y_score = classifier.predict(X_test.T)
    
    tn, fp, fn, tp = confusion_matrix(Y_test.T, y_score).ravel()
    precission = tp/(tp+fp)
    sensitivity = tp/(tp+fn) #RECALL
    specificity = tn/(tn+fp)
    f1score = 2*(precission*sensitivity)/(precission+sensitivity)
    
    return sensitivity, specificity, precission


# 4. G A U S S I A N   P R O C E S S   M O D E L
#===============================================
def gaussian_pro(X_train,Y_train,X_test,Y_test):
    kernel = 1.0 * RBF([1.0]*7)
    clf = GaussianProcessClassifier(kernel=kernel).fit(X_train.T, Y_train.T)
    
    y_score =  clf.predict(X_test.T)
    
    
    
    tn, fp, fn, tp = confusion_matrix(Y_test.T, y_score).ravel()
    precission = tp/(tp+fp)
    sensitivity = tp/(tp+fn) #RECALL
    specificity = tn/(tn+fp)
    f1score = 2*(precission*sensitivity)/(precission+sensitivity)
    
    return sensitivity, specificity, precission

# 5.  R A N D O M  G U E S S I N G
#=================================
def random_guess(Y_test):
    y_guess = np.random.rand(Y_test.shape[0],Y_test.shape[1])
    y_guess[y_guess>0.5] = 1
    y_guess[y_guess<=0.5] = 0
    
    
    tn, fp, fn, tp = confusion_matrix(Y_test.T, y_guess.T).ravel()
    precission = tp/(tp+fp)
    sensitivity = tp/(tp+fn) #RECALL
    specificity = tn/(tn+fp)
    return sensitivity, specificity,precission
