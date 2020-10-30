import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import request
import json


train=pd.read_csv('Admission_Predict.csv')
train.rename(columns=lambda x: x.replace('DEATH_EVENT', 'Admission'), inplace=True)

train["gre"] = train["gre"]
bins = [39, 50, 65, 70, np.inf]
bins = [289, 304, 312, 317, 323, 328, np.inf]
labels = ['289-304', '304-312', '312-317', '317-323', '323-328', '328-340']
train['ClassesGRE'] = pd.cut(train["gre"], bins, labels = labels)

def prediction(param):

    param=np.array(param).reshape(1,-1) 
    cls=pickle.load(open("cls_Admission.pkl", "rb"))
    return (cls.predict(param))

def entrainement():
    predictors = train.drop(['Admission'], axis=1)
    target = train['Admission']
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
    cls=RandomForestClassifier(max_depth=12,n_estimators=300).fit(x_train,y_train)
    filename = 'cls_Admission.pkl'
    pickle.dump(cls, open(filename, 'wb'))
    return(cls.score(x_val,y_val))

def entrainement():

    predictors = train.drop(['Mort'], axis=1)
    target = train["Mort"]

    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
    cls=RandomForestClassifier(max_depth=12,n_estimators=300).fit(x_train,y_train)
    
    #sauver cls
    filename = 'cls_heart_attack.pkl'
    pickle.dump(cls, open(filename, 'wb'))

    return(cls.score(x_val,y_val))