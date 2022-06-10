# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:28:17 2021

@author: saufiero
"""

# Load libraries
from tensorflow.keras.models import load_model
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns

# The model requires a three-dimensional input with [samples, time steps, features].
# The features are the 12-lead ECGs in the following order:
# I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.
# A test example x and y are given

# Load test data xTest and yTest
filepath = os.path.join('.',"xTest.pckl")
with open(filepath, 'rb') as f:
    xTest = pickle.load(f)
filepath = os.path.join('.',"yTest.pckl")
with open(filepath, 'rb') as f:
    yTest = pickle.load(f)
    
# Load model
modelPath= os.path.join('bestModels', 'allECGs','lqts1', 'fold4.h5')
model = load_model(modelPath)
# predict label
yPred=(model.predict(xTest)>0.5).astype("int32") 
# or to get probability score run
#yPred=model.predict(xTest) 

# Confusion matrix
cm=confusion_matrix(yTest, yPred, labels=[0,1])
sns.heatmap(cm, annot=True, annot_kws={"size": 16},fmt='g',cmap="Blues") 


