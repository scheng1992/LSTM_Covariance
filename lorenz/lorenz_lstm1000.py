# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:39:58 2021

@author: siboc
"""

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import tensorflow as tf
import keras.backend as K
import sys
# check scikit-learn version

# check scikit-learn version
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def data_set_order(file):
	train_data = np.array(pd.read_csv(file))[:-2,:]
	r0=train_data[:,:1001]
	r1=train_data[:,1001:2002]
	r2=train_data[:,2002:3003]
	r3=train_data[:,3003:]
	r3[:,-1]=r3[:,-1]/100
	train_data=np.insert(r0,[i+1 for i in range(r0.shape[1])],r1,axis=1)
	train_data=np.insert(train_data,[(i+1)*2 for i in range(int(train_data.shape[1]/2))],r2,axis=1)
	train_data=np.concatenate((train_data,r3),axis=1)
	return train_data
#input
train_data = data_set_order('lorenz_cov_train_v2/trainset_withx_steps1000_11.csv')
print("train_data shape: ",train_data.shape)

print(f"training dataset size: {train_data.shape[0]*0.9}")
print(f"validation dataset size: {train_data.shape[0]*0.1}")

sys.exit()

# train_data1 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis3.csv')
# print("train_data1 shape: ",train_data1.shape)

# train_data2 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis4.csv')
# print("train_data2 shape: ",train_data2.shape)

# train_data3 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis5.csv')
# print("train_data3 shape: ",train_data3.shape)

# train_data4 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis6.csv')
# print("train_data4 shape: ",train_data4.shape)

# train_data5 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis7.csv')
# print("train_data5 shape: ",train_data5.shape)

# train_data6 = data_set_order('lorenz_cov_train/trainset_withx_repeat10bis8.csv')
# print("train_data6 shape: ",train_data6.shape)

#size: num_steps*3,r1,r2,r3,v


#########################################################################################

#train_data = np.array(pd.read_csv('data_1000steps/trainset_withx_1000steps.csv'))
#
#
#train_data1 = np.array(pd.read_csv('data_1000steps/trainset_withx_1000stepsbis1.csv'))
#
#train_data2 = np.array(pd.read_csv('data_1000steps/trainset_withx_1000stepsbis2.csv'))
#
#train_data3 = np.array(pd.read_csv('data_1000steps/trainset_withx_1000stepsbis3.csv'))





# train_data = np.concatenate((train_data6,train_data5),axis = 0)

# train_data = np.concatenate((train_data,train_data4),axis = 0)

# train_data = np.concatenate((train_data,train_data3),axis = 0)

# train_data = np.concatenate((train_data,train_data2),axis = 0)

# train_data = np.concatenate((train_data,train_data1),axis = 0)

# train_data = np.concatenate((train_data,train_data0),axis = 0)

# train_data=train_data[:120000,:]


#weightstrain_data[:,604:]
np.random.shuffle(train_data )

input_data = train_data[:,0:3003]
output_data = train_data[:,3003:]



########################################################################
train_part = 0.97

threshold = int(train_part*train_data.shape[0])


##########################################################################

train_input = input_data[:threshold]

print("input_data shape: ",input_data.shape)

train_output = output_data[:threshold]

print("output_data shape: ",output_data.shape)


test_input = input_data [threshold:]

true_test_output = output_data[threshold:]



X1 = train_input
Y1 = train_output

X2 = test_input
#Y2 = ValidationSet_Y

############################################################################



#def my_loss_fn(y_true, y_pred):
#    
#    return K.mean(K.abs(y_true - y_pred) * weight)

# ========================================================================================
from keras.layers import LSTM,Dropout
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam




# save data
import os
import json
import pickle

if not os.path.isdir('save_data_v2'):
	os.makedirs('save_data_v2')

hidden_size=200


input_sample=input_data.shape[0]  #for one sample

output_sample=output_data.shape[0]

input_data=input_data.reshape(input_sample,1001,3)  #201 is the time steps in data_generation
output_data=output_data.reshape(output_sample,4)

use_dropout=True
model = Sequential()
model.add(LSTM(hidden_size,input_shape=(1001,3)))

model.add(Dense(4))
# opt = Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
print(model.summary())

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
# modelcheckpoint
mc=ModelCheckpoint('save_data_v2/sequentiallstm1000_ing.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

history=model.fit(input_data, output_data, validation_split=0.1, epochs=100, batch_size=128, verbose=1,callbacks=[es,mc])



# model.save('save_data/sequentiallstm2')
model.save('save_data_v2/sequentiallstm1000_ing_f.h5')

# https://stackoverflow.com/a/44674337/10349608
with open('save_data_v2/sequentiallstm1000_ing_history.pickle', 'wb') as file_his:
	pickle.dump(history.history, file_his)


# Calculate predictions
PredTestSet = model.predict(X1.reshape(X1.shape[0],1001,3))
PredValSet = model.predict(X2.reshape(X2.shape[0],1001,3))




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.savefig('figure_dp/loss_trace.eps', format='eps',bbox_inches='tight')
plt.show()



plt.plot(PredValSet[:,2],true_test_output[:,2],'o', color='blue',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.show()

plt.plot(PredValSet[:,3],true_test_output[:,3],'o',color='blue',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.show()



# predint = model.predict(train_input[:3000])

# trueint = train_output[:3000]


# plt.plot(predint[:,3],trueint[:,3],'o', color='blue',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.show()

