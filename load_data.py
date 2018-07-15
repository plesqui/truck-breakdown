import math
import pandas as pd
import numpy as np

 
 # Loading the dataset:
def load_data_oil_old(filename, t = 14):
"""
Inputs 
filename = string path to file
t = time until breakdown in days (2 weeks default)

Outputs
X_train = array of shape (n_features, m) where n_features is the number of features (7) and m is the number of training examples
Y_train = array of shape (1, m), representing the probability (0 or 1) of breakdown within t days for each m training example.
X_test, Y_test = arrays representing the test set 
"""
    df_data = pd.read_csv(filename)

    # Which sensors do you want to train on?
    sensors = ['oilhours','V100','Fe', 'Cu', 'Al', 'Mo','Sulf']
    # Number of features
    n_feat = len(sensors)

    # Clear data a little bit (drop NaN):
    df_data = df_data[df_data['compart'] == 'Engine'] # Select Engine only
    df_data = df_data.dropna(subset = ['DamageDelta'], axis = 0)

    
    for z in sensors:
        df_data[z] = pd.to_numeric(df_data[z], errors='coerce')
        df_data[z] = df_data[z].fillna(value = np.mean(df_data[z]))


    # Convert into classifiers [2 weeks interval!]
    

    df_data['DamageDelta'][df_data['DamageDelta']<=t] = 1
    df_data['DamageDelta'][df_data['DamageDelta']>t] = 0


    # Shuffe the data:
    df_data = df_data.sample(frac = 1)

    # Fix the column 'V100', and remove the string '-'
    df_data['V100'] = pd.to_numeric(df_data['V100'], errors='coerce')
    df_data['V100'] = df_data['V100'].fillna(value = np.mean(df_data['V100']))

    # Number of total training examples

    m = df_data.shape[0]

    # Number of trianing and test

    frac_train = 0.7
    m_train = np.int(frac_train*m)
    m_test = m-m_train
    
    
    # Assign X and Y
    X = df_data[sensors]
    Y = df_data['DamageDelta']
    
    
    # Convert to numpy array
    X = np.array(X)
    Y = np.array(Y)
    Y = np.reshape(Y, (X.shape[0],1))

    # Dividing training for even positives and negatives
    
    XY = np.column_stack((X,Y))
    
    XY = XY[np.argsort(XY[:,len(sensors)]),:]
    print(np.sum(Y))
    XYtr = np.zeros((1,len(sensors)+1))
    XYts = np.zeros((1,len(sensors)+1))
    nT = 30 #total training points
    n = nT / 2
    for c in np.unique(XY[:,len(sensors)]):
            v0 = np.where(XY[:,len(sensors)]==c)[0]
            min_ = np.min(v0)
            max_ = np.max(v0)
            v = np.linspace(min_,max_,num=n, dtype = int)
            XYtr = np.vstack((XYtr,XY[v,:]))
            vt = []
            for i in range(min_,max_+1):
                if len(np.where(v == i)[0]) < 1:
                    vt.append(i)
            XYts = np.vstack((XYts,XY[vt,:]))
    XYtr = XYtr[1:,:]
    XYts = XYts[1:,:]
    
    for i,xy in enumerate(XYtr):
        XYtr[i,:len(sensors)] = np.divide(XYtr[i,:len(sensors)],XYtr[i,0])
    
    for i,xy, in enumerate(XYts):
        XYts[i,:len(sensors)] = np.divide(XYts[i,:len(sensors)],XYts[i,0])
        
    
    # Normalize the data:
    X_normalized_tr = preprocessing.scale(XYtr[:,:len(sensors)])
    X_normalized_ts = preprocessing.scale(XYts[:,:len(sensors)])

    # Bring to the shape I like
    X_train = X_normalized_tr.T
    X_test = X_normalized_ts.T
    Y_train = np.atleast_2d(XYtr[:,len(sensors)])
    Y_test = np.atleast_2d(XYts[:,len(sensors)])

    # Check shapes:

    print("Number of positive examples in Training: "+str(np.sum(Y_train)))
    print("Number of negative examples in Training: "+str(Y_train.shape[1]-np.sum(Y_train)))
    
    return X_train, Y_train, X_test, Y_test
