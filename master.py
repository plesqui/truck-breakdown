# Master script that performs the following operations:
# 1) Load the training and test set
# 2) Train the four different models included in models.py
# 3) Evaluate the sensitivity, specificity and precission of each model.
#The assessment is done for different predicting time intervals, in days.

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

%matplotlib inline


file = r'OilAnalysis_vs_DaystoFailure.csv' # Using only Hitachi Trucks, and Engine-oil suspected faults

days = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]


rep = 10
NN_sensitivity = np.zeros((len(days),rep))
NN_specificity = np.zeros((len(days),rep))
NN_precission = np.zeros((len(days),rep))

SVM_sensitivity = np.zeros((len(days),rep))
SVM_specificity = np.zeros((len(days),rep))
SVM_precission = np.zeros((len(days),rep))


RF_sensitivity = np.zeros((len(days),rep))
RF_specificity = np.zeros((len(days),rep))
RF_precission = np.zeros((len(days),rep))

GS_sensitivity = np.zeros((len(days),rep))
GS_specificity = np.zeros((len(days),rep))
GS_precission = np.zeros((len(days),rep))


RR_sensitivity = np.zeros((len(days),rep))
RR_specificity = np.zeros((len(days),rep))
RR_precission = np.zeros((len(days),rep))

k =0
for i in days:
    
    print('Time Window = ', i,' days')
    for j in range(0,rep):
        X_train,Y_train,X_test,Y_test = load_data_oil_old(file, i)
        _,NN_sensitivity[k,j],NN_specificity[k,j],NN_precission[k,j] = model(X_train,Y_train,X_test,Y_test)
        SVM_sensitivity[k,j],SVM_specificity[k,j],SVM_precission[k,j] = run_svm(X_train,Y_train,X_test,Y_test)
        RF_sensitivity[k,j],RF_specificity[k,j],RF_precission[k,j] = run_forest(X_train,Y_train,X_test,Y_test)
        GS_sensitivity[k,j],GS_specificity[k,j],GS_precission[k,j] = gaussian_pro(X_train,Y_train,X_test,Y_test)
        RR_sensitivity[k,j],RR_specificity[k,j],RR_precission[k,j] = random_guess(Y_test)
    k += 1
    
# Plot the results
NN_f1score = np.zeros((len(days),1))
SVM_f1score = np.zeros((len(days),1))
RF_f1score = np.zeros((len(days),1))
GS_f1score = np.zeros((len(days),1))
RR_f1score = np.zeros((len(days),1))


k = 0
for i in days:
    
    NN_f1score[k] = 2*(np.squeeze(np.mean(NN_precission, axis = 1))[k]*np.squeeze(np.mean(NN_sensitivity, axis = 1)[k]))/(np.squeeze(np.mean(NN_precission, axis = 1))[k]+np.squeeze(np.mean(NN_sensitivity, axis = 1)[k]))
    SVM_f1score[k] = 2*(np.squeeze(np.mean(SVM_precission, axis = 1))[k]*np.squeeze(np.mean(SVM_sensitivity, axis = 1)[k]))/(np.squeeze(np.mean(SVM_precission, axis = 1))[k]+np.squeeze(np.mean(SVM_sensitivity, axis = 1)[k]))
    RF_f1score[k] = 2*(np.squeeze(np.mean(RF_precission, axis = 1))[k]*np.squeeze(np.mean(RF_sensitivity, axis = 1)[k]))/(np.squeeze(np.mean(RF_precission, axis = 1))[k]+np.squeeze(np.mean(RF_sensitivity, axis = 1)[k]))
    GS_f1score[k] = 2*(np.squeeze(np.mean(GS_precission, axis = 1))[k]*np.squeeze(np.mean(GS_sensitivity, axis = 1)[k]))/(np.squeeze(np.mean(GS_precission, axis = 1))[k]+np.squeeze(np.mean(GS_sensitivity, axis = 1)[k]))
    RR_f1score[k] = 2*(np.squeeze(np.mean(RR_precission, axis = 1))[k]*np.squeeze(np.mean(RR_sensitivity, axis = 1)[k]))/(np.squeeze(np.mean(RR_precission, axis = 1))[k]+np.squeeze(np.mean(RR_sensitivity, axis = 1)[k]))
    k += 1
    
NN_f1score = np.nan_to_num(NN_f1score)
SVM_f1score = np.nan_to_num(SVM_f1score)
RF_f1score = np.nan_to_num(RF_f1score)
GS_f1score = np.nan_to_num(GS_f1score)
RR_f1score = np.nan_to_num(RR_f1score)


plt.figure(figsize=(18,10))
plt.subplot(2,3,1)
plt.plot(np.array(days),np.squeeze(np.mean(NN_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(NN_specificity, axis = 1)),
         np.array(days),np.squeeze(np.mean(NN_f1score, axis = 1)))
plt.ylabel('Metric')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.title("For NebbieNET")
plt.legend(["SENSITIVITY","SPECIFICITY","F1-SCORE"])

plt.subplot(2,3,2)
plt.plot(np.array(days),np.squeeze(np.mean(SVM_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(SVM_specificity, axis = 1)),
         np.array(days),np.squeeze(np.mean(SVM_f1score, axis = 1)))
plt.ylabel('Metric')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.title("For SVMLinear") 
plt.legend(["SENSITIVITY","SPECIFICITY","F1-SCORE"])


plt.subplot(2,3,3)
plt.plot(np.array(days),np.squeeze(np.mean(RF_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RF_specificity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RF_f1score, axis = 1)))
plt.ylabel('Metric')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.title("For RandomForest") 
plt.legend(["SENSITIVITY","SPECIFICITY","F1-SCORE"])



plt.subplot(2,3,5)
plt.plot(np.array(days),np.squeeze(np.mean(RR_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RR_specificity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RR_f1score, axis = 1)))
plt.ylabel('Metric')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.title("For RandomGuess") 
plt.legend(["SENSITIVITY","SPECIFICITY","F1-SCORE"])


plt.subplot(2,3,4)
plt.plot(np.array(days),np.squeeze(np.mean(GS_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(GS_specificity, axis = 1)),
         np.array(days),np.squeeze(np.mean(GS_f1score, axis = 1)))
plt.ylabel('Metric')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.title("For GaussianProcess") 
plt.legend(["SENSITIVITY","SPECIFICITY","F1-SCORE"])



plt.show()

# Plot a comparison of the models
plt.plot(np.array(days),np.squeeze(np.mean(NN_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(SVM_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RF_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(GS_sensitivity, axis = 1)),
         np.array(days),np.squeeze(np.mean(RR_sensitivity, axis = 1)))
plt.ylabel('SENSITIVITY')
plt.xlabel('TIME WINDOW')
plt.ylim((0,1))
plt.legend(['NebbieNet', 'SVM', 'RandomForest', 'GaussianProcess','RandomGuess'])
