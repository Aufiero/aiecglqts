# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:43:33 2021

@author: saufiero
"""

# Import libraries 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LeakyReLU, BatchNormalization, MaxPooling1D
from tensorflow.keras.metrics import Recall



def get1DCNN(xTrain,
             nClasses,
             loss,
             activationDense): 

    
    """ The function is used to build the 1DCNN architecture used to train the model.
    
    param: xTrain This is 3D array of n samples, 2500 timesteps, and 12 feature (12-lead ECG)
    param: nClasses This is an integer specifying the number of classes in the last Dense layer.
    param: loss This is a string specifying the loss function to use in the compile layer.
    param: activationDense This is a string specifying the activation function to use in the last Dense layer.
    
    """ 
    nTimesteps, nFeatures = xTrain.shape[1], xTrain.shape[2]
   
    # Feature Extraction
    classifier = Sequential()

    classifier.add(Conv1D(filters=12,input_shape=(nTimesteps,nFeatures),kernel_size = 3, strides = 1,padding='same')) 
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())

    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 

    classifier.add(Conv1D(filters=12,kernel_size = 3, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 
  
    classifier.add(Conv1D(filters=12,kernel_size = 3, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 
    
    classifier.add(Conv1D(filters=12,kernel_size = 3, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25))


    classifier.add(Conv1D(filters=12,kernel_size = 3, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25))
    
     
    classifier.add(Conv1D(filters=12,kernel_size = 5, strides = 1,padding='same')) 
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv1D(filters=12,kernel_size = 5, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 
    
    classifier.add(Conv1D(filters=12,kernel_size = 5, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 
    
    classifier.add(Conv1D(filters=12,kernel_size = 5, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25)) 
    
    classifier.add(Conv1D(filters=12,kernel_size = 5, strides = 1,padding='same'))
    classifier.add(LeakyReLU(alpha=0.3))
    classifier.add(BatchNormalization())
    
    classifier.add(MaxPooling1D(pool_size= 2, strides=2,padding='same'))
    classifier.add(Dropout(0.25))
 
    classifier.add(Flatten())
    
    # Classification 
    classifier.add(Dense(units = nClasses, activation=activationDense))
    recall = Recall(name='recall')

    classifier.compile(loss=loss, optimizer='adam', metrics=['accuracy',recall])
    
    return classifier

