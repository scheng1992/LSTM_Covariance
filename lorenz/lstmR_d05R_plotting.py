# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:39:58 2021

@author: siboc
"""

import numpy as np

import matplotlib.pyplot as plt



# check scikit-learn version

# check scikit-learn version

import pandas as pd

# def data_set_order(file):
#     train_data = np.array(pd.read_csv(file))
#     r0=train_data[:,:1001]
#     r1=train_data[:,1001:2002]
#     r2=train_data[:,2002:3003]
#     r3=train_data[:,3003:]/10
#     train_data=np.insert(r0,[i+1 for i in range(r0.shape[1])],r1,axis=1)
#     train_data=np.insert(train_data,[(i+1)*2 for i in range(int(train_data.shape[1]/2))],r2,axis=1)
#     train_data=np.concatenate((train_data,r3),axis=1)
#     return train_data

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

# def data_set_order(file):
#  	train_data = np.array(pd.read_csv(file))[:-2,:]
#  	r0=train_data[:,:1001][:,:201]
#  	r1=train_data[:,1001:2002][:,:201]
#  	r2=train_data[:,2002:3003][:,:201]
#  	r3=train_data[:,3003:]
#  	r3[:,-1]=r3[:,-1]/100
#  	train_data=np.insert(r0,[i+1 for i in range(r0.shape[1])],r1,axis=1)
#  	train_data=np.insert(train_data,[(i+1)*2 for i in range(int(train_data.shape[1]/2))],r2,axis=1)
#  	train_data=np.concatenate((train_data,r3),axis=1)
#  	return train_data

train_data = data_set_order('lorenz_cov_train_v2/trainset_withx_steps1000_11.csv')[:10000,:]




print("train_data shape: ",train_data.shape)



# ###############################################################################

# LSTM1000
input_data = train_data[:,0:1001*3]
output_data = train_data[:,1001*3:]

# LSTM200
# input_data = train_data[:,0:603]
# output_data = train_data[:,603:]



########################################################################
train_part = 0.97

# threshold = int(train_part*train_data.shape[0])
threshold=10000


##########################################################################

test_input = input_data[:threshold,:]

true_test_output = output_data[:threshold,:]

# test_input = input_data [threshold:,:]

# true_test_output = output_data[threshold:,:]



# X1 = train_input
# Y1 = train_output

X2 = test_input

print("X2 shape: ",X2.shape[0])
#Y2 = ValidationSet_Y



############################################################################

R=np.load('label_data/di05_original_version_R_all_10000.npy')
PredValSet2=np.zeros((R.shape[0],4))

for i in range(R.shape[0]):
    print(i)
    r0=R[i,0,1]
    r1=R[i,0,2]
    r2=R[i,1,2]
    r3=np.trace(R[i,:,:])/3
    r=np.array([[r0,r1,r2,r3]])
    
    if i==0:
        PredValSet2=r.copy()
    else:
        PredValSet2=np.concatenate((PredValSet2,r),axis = 0)


#def my_loss_fn(y_true, y_pred):
#    
#    return K.mean(K.abs(y_true - y_pred) * weight)

# from tensorflow.keras.models import load_model


# model1=load_model('data2/sequentiallstm1000_ing_f.h5')
# model1=load_model('data2/sequentiallstm200_ing_f.h5')


# Calculate predictions


# PredValSet2 = model1.predict(X2.reshape(X2.shape[0],1001,3))
# PredValSet2 = model1.predict(X2.reshape(X2.shape[0],201,3))

# PredTestSet = model.predict(X1)
# PredValSet = model.predict(X2)

# Save predictions
#np.savetxt("numerique/trainresults_raindebit.csv", PredTestSet, delimiter=",")
#np.savetxt("numerique/valresults_raindebit.csv", PredValSet, delimiter=",")


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# #plt.savefig('figure_dp/loss_trace.eps', format='eps',bbox_inches='tight')
# plt.show()

#plt.plot(true_test_output[:,1],'r',label = "true")
#plt.plot(PredValSet[:,1],label = "model")
#plt.title("1st coeff linear")
#plt.legend()
#plt.show()

#deep_error = []
#
#for i in range(150):
#    
#    deep_error.append(np.linalg.norm(PredValSet[:,i]-true_test_output[:,i]))
#
#print('deep_error',deep_error)

# %%
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.plot(PredValSet1[:,0],true_test_output[:,0],'o', color='blue',markersize=5,label='lstm')


# %%

plt.plot(PredValSet2[:,0]/PredValSet2[:,3],true_test_output[:,0],'o', color='b',markersize=5)
plt.plot(true_test_output[:,0],true_test_output[:,0],'-', color='r',linewidth=5)

plt.xlabel('prediction',fontsize=22)
plt.ylabel('true value',fontsize=22)
plt.xticks([-2.00,-1.00,0.00,1.00,2.00])
plt.yticks([-1.00,-0.50,0.00,0.50,1.00])
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.legend()
plt.show()


# %%


# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.plot(PredValSet1[:,0],true_test_output[:,0],'o', color='blue',markersize=5,label='lstm')
# plt.plot(PredValSet2[:,0],true_test_output[:,0],'o', color='b',markersize=5)
# plt.plot(true_test_output[:,0],true_test_output[:,0],'o', color='r',markersize=5)
# plt.plot(PredValSet3[:,0],true_test_output[:,0],'o', color='green',markersize=5)
# plt.plot(PredValSet4[:,0],true_test_output[:,0],'o', color='c',markersize=5)
# plt.plot(PredValSet5[:,0],true_test_output[:,0],'o', color='m',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.legend()
# plt.show()


# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.plot(PredValSet1[:,1],true_test_output[:,1],'o', color='blue',markersize=5,label='lstm')
# %%
plt.plot(PredValSet2[:,1]/PredValSet2[:,3],true_test_output[:,1],'o', color='b',markersize=5)
plt.plot(true_test_output[:,1],true_test_output[:,1],'-', color='r',linewidth=5)
# plt.xlabel('prediction',fontsize=22)
# plt.ylabel('true value',fontsize=22)
# plt.xticks([-1.00,-0.50,0.00,0.50,1.00])
# plt.yticks([-1.00,-0.50,0.00,0.50,1.00])
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.plot(PredValSet3[:,1],true_test_output[:,1],'o', color='green',markersize=5)
# plt.plot(PredValSet4[:,1],true_test_output[:,1],'o', color='c',markersize=5)
# plt.plot(PredValSet5[:,1],true_test_output[:,1],'o', color='m',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.legend()
plt.show()
# %%

# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.plot(PredValSet1[:,1],true_test_output[:,1],'o', color='blue',markersize=5,label='lstm')
# plt.plot(PredValSet2[:,1],true_test_output[:,1],'o', color='b',markersize=5)
# plt.plot(true_test_output[:,1],true_test_output[:,1],'o', color='r',markersize=5)
# plt.plot(PredValSet3[:,1],true_test_output[:,1],'o', color='green',markersize=5)
# plt.plot(PredValSet4[:,1],true_test_output[:,1],'o', color='c',markersize=5)
# plt.plot(PredValSet5[:,1],true_test_output[:,1],'o', color='m',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.legend()
# plt.show()

# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.plot(PredValSet1[:,2],true_test_output[:,2],'o', color='blue',markersize=5,label='lstm')
# plt.plot(PredValSet2[:,2],true_test_output[:,2],'o', color='b',markersize=5)
# plt.plot(true_test_output[:,2],true_test_output[:,2],'o', color='r',markersize=5)
# plt.plot(PredValSet3[:,2],true_test_output[:,2],'o', color='green',markersize=5)
# plt.plot(PredValSet4[:,2],true_test_output[:,2],'o', color='c',markersize=5)
# plt.plot(PredValSet5[:,2],true_test_output[:,2],'o', color='m',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.legend()
# plt.show()

# plt.plot(PredValSet1[:,2],true_test_output[:,2],'o', color='blue',markersize=5,label='lstm')
# %%
plt.plot(PredValSet2[:,2]/PredValSet2[:,3],true_test_output[:,2],'o', color='b',markersize=5)
plt.plot(true_test_output[:,2],true_test_output[:,2],'-', color='r',linewidth=5)
plt.xlabel('prediction',fontsize=22)
plt.ylabel('true value',fontsize=22)
plt.xticks([-1.00,-0.50,0.00,0.50,1.00])
plt.yticks([-1.00,-0.50,0.00,0.50,1.00])
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.show()
# %%

# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.plot(PredValSet1[:,3],true_test_output[:,3],'o',color='blue',markersize=5,label='lstm')
# plt.plot(PredValSet2[:,3],true_test_output[:,3],'o', color='b',markersize=5)
# plt.plot(true_test_output[:,3],true_test_output[:,3],'o', color='r',markersize=5)
# plt.plot(PredValSet3[:,3],true_test_output[:,3],'o', color='green',markersize=5)
# plt.plot(PredValSet4[:,3],true_test_output[:,3],'o', color='c',markersize=5)
# plt.plot(PredValSet5[:,3],true_test_output[:,3],'o', color='m',markersize=5)
# plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.legend()
# plt.show()

# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.plot(PredValSet1[:,3],true_test_output[:,3],'o',color='blue',markersize=5,label='lstm')

# %%
plt.plot(PredValSet2[:,3],true_test_output[:,3]*100,'o', color='b',markersize=5)
plt.plot(true_test_output[:,3]*100,true_test_output[:,3]*100,'-', color='r',linewidth=5)
# plt.xlabel('prediction',fontsize=22)
# plt.ylabel('true value',fontsize=22)
# # plt.xticks([-1.00,-0.50,0.00,0.50,1.00])
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)

plt.legend()
plt.show()
# %%

# plt.plot(true_test_output[:,0],color='b',label='r0')
# plt.plot(true_test_output[:,1],color='r',label='r1')
# plt.plot(true_test_output[:,2],color='y',label='r2')
# plt.legend()
# plt.show()



##########################################################################################"

# predint = model.predict(train_input[:3000])

# trueint = train_output[:3000]


# plt.plot(predint[:,3],trueint[:,3],'o', color='blue',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# plt.show()

