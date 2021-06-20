# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:39:58 2021

@author: siboc
"""

import numpy as np
import scipy
import math



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import tensorflow as tf
import tensorflow.keras.backend as K
# check scikit-learn version

# check scikit-learn version
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


# import keras

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import json
import pickle




# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0:3], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
#=======================================================================
                        # Generator

class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for keras
    # list_IDs: all IDs/all files
    # list_IDs_temp: studying batch IDs
    def __init__(self, list_IDs,batch_size=1,dim=(1000,200),n_channels=1, shuffle=True):

        self.dim=dim
        self.batch_size=batch_size
        # self.labels=labels
        self.list_IDs=list_IDs
        self.n_channels=n_channels
        # self.n_classes=n_classes
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):

        print(index)
        
        # we do not know the whole length of the matrix in a file without loading all of the data ??????????
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X,y

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        
        for i,ID in enumerate(list_IDs_temp):
            # open the file
            obs = np.load('uniform020/trainset_withx_repeat_shwater3_uniform0011_test6_'+str(ID)+'.npy')
            
            obs_size=obs.shape[0]

            print("ID: ",ID)

            print("obs_size: ",obs_size)

            X=np.empty((obs_size,200*1000))
            y=np.empty((obs_size,101))

            y=obs[:,-101:]
            y[:,-101:-1]=y[:,-101:-1]*1000
            y[:,-1]=y[:,-1]/8

            X=obs[:,:200*1000]
            X=X.reshape((obs_size,200,1000))[:,:,:200]

            X=np.array([X[j].transpose() for j in range(obs_size)])


            

        
        return X,y

print("31")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
# #====================== Read file list_IDs =================================================

partition=np.array([i for i in range(211)])

# #====================== Parameters =================================================
Params={'dim':(1000,200),
        'batch_size':1,
        'n_channels':1,
        'shuffle':True}


# train_data1000 = np.array(pd.read_csv('0001D07/trainset_withx_repeat_shwater3_0001D07_total_3.csv',delimiter=",", 
#                  header=None, 
#                  index_col=False))

# obs = train_data1000.reshape((train_data1000.shape[0],200*1000+101))

# X=np.empty((train_data1000.shape[0],200*1000))
# y=np.empty((train_data1000.shape[0],101))

# y=obs[:,-101:]
# X=obs[:,:200*1000].reshape((train_data1000.shape[0],200,1000))
# X=np.array([X[i].transpose() for i in range(X.shape[0])])


# input_data=X[:,:,:200]
# output_data=y



# #====================== Generators =================================================
train_part = 0.97
threshold = int(train_part*len(partition))

# input_generator=DataGenerator(partition,**Params)

training_generator=DataGenerator(partition[:threshold],**Params)
validation_generator=DataGenerator(partition[threshold:],**Params)



#============================= Model Design ==========================================


if not os.path.isdir('data2'):
    os.makedirs('data2')

# try:

hidden_size=200

model = Sequential()
model.add(LSTM(hidden_size,input_shape=(200,200)))
model.add(Dense(101))

# model=load_model('data2/sequentiallstm2222200_b128_h200_norm_out_gen22.h5',compile = False)

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])

print(model.summary())

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=100)
# modelcheckpoint
mc=ModelCheckpoint('data2/sequentiallstm2222200_b128_h200_norm_out_gen22.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

# history=model.fit(input_data, output_data, validation_split=train_part, epochs=100, batch_size=128, verbose=1,callbacks=[es,mc])
history=model.fit(x=training_generator, validation_data=validation_generator, epochs=1000, validation_batch_size=5,verbose=1, callbacks=[es,mc])
# history=model.fit(x=training_generator, validation_data=validation_generator, epochs=1000, use_multiprocessing=True, workers=6,verbose=1,callbacks=[es,mc])




# model.save('save_data/sequentiallstm2')
model.save('data2/sequentiallstm2222200_b128_h200_norm_out_gen22_f.h5')

# https://stackoverflow.com/a/44674337/10349608
with open('data2/sequentiallstm2222200_b128_h200_norm_out_gen22_history.pickle', 'wb') as file_his:
    pickle.dump(history.history, file_his)



