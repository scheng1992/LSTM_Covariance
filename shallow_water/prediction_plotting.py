# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:39:58 2021

@author: siboc
"""

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

#


data=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1114.npy').astype(np.float32)
data1=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_948.npy').astype(np.float32)
train_data=np.concatenate((data,data1),axis=0)
del data
del data1

data2=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_947.npy').astype(np.float32)
train_data=np.concatenate((train_data,data2),axis=0)
del data2

data3=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_940.npy').astype(np.float32)
train_data=np.concatenate((train_data,data3),axis=0)
del data3

data4=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_941.npy').astype(np.float32)
train_data=np.concatenate((train_data,data4),axis=0)
del data4

data5=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1111.npy').astype(np.float32)
train_data=np.concatenate((train_data,data5),axis=0)
del data5

data6=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1112.npy').astype(np.float32)
train_data=np.concatenate((train_data,data6),axis=0)
del data6

data7=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_946.npy').astype(np.float32)
train_data=np.concatenate((train_data,data7),axis=0)
del data7

data8=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_945.npy').astype(np.float32)
train_data=np.concatenate((train_data,data8),axis=0)
del data8

data9=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_944.npy').astype(np.float32)
train_data=np.concatenate((train_data,data9),axis=0)
del data9

data10=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_943.npy').astype(np.float32)
train_data=np.concatenate((train_data,data10),axis=0)
del data10

data11=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_942.npy').astype(np.float32)
train_data=np.concatenate((train_data,data11),axis=0)
del data11

data12=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1116.npy').astype(np.float32)
train_data=np.concatenate((train_data,data12),axis=0)
del data12

data13=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1115.npy').astype(np.float32)
train_data=np.concatenate((train_data,data13),axis=0)
del data13

data14=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1113.npy').astype(np.float32)
train_data=np.concatenate((train_data,data14),axis=0)
del data14

data15=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_939.npy').astype(np.float32)
train_data=np.concatenate((train_data,data15),axis=0)
del data15

data16=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_938.npy').astype(np.float32)
train_data=np.concatenate((train_data,data16),axis=0)
del data16

data17=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_937.npy').astype(np.float32)
train_data=np.concatenate((train_data,data17),axis=0)
del data17

data18=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1117.npy').astype(np.float32)
train_data=np.concatenate((train_data,data18),axis=0)
del data18

data19=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1118.npy').astype(np.float32)
train_data=np.concatenate((train_data,data19),axis=0)
del data19

# train_data=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_1179.npy').astype(np.float32) no
# train_data=np.load('data2/trainset_withx_repeat_shwater3_uniform0011_test6_944.npy').astype(np.float32) 

# %%

print("train_data shape: ",train_data.shape)
# ###############################################################################

obs = train_data.reshape((train_data.shape[0],200*1000+101))
del train_data

# %%
X=None
y=None

y=obs[:,-101:]
y[:,-101:-1]=y[:,-101:-1]*1000
y[:,-1]=y[:,-1]/8

y=y.reshape((obs.shape[0],101))

X=obs[:,:200*1000].reshape((obs.shape[0],200,1000))
# X=std_scaler.transform(X)

X=np.array([X[i].transpose() for i in range(X.shape[0])])

# input_data=X[:,:1000,:]
input_data=X[:,:200,:]
output_data=y

##########################################################################

train_part = 0.97

threshold = int(train_part*obs.shape[0])


##########################################################################
# train_input = input_data[:threshold,:]

# train_output = output_data[:threshold,:]

# test_input = input_data [threshold:,:]

# true_test_output = output_data[threshold:,:]



# X1 = train_input
# Y1 = train_output

X2 = input_data

true_test_output=output_data


# train_input = input_data[:threshold,:]

# train_output = output_data[:threshold,:]


# test_input=input_data


# true_test_output = output_data



# X1 = train_input
# Y1 = train_output

# X2 = test_input


############################################################################



#def my_loss_fn(y_true, y_pred):
#    
#    return K.mean(K.abs(y_true - y_pred) * weight)

# %%
from tensorflow.keras.models import load_model

# model1=load_model('data2/sequentiallstm200_b128_h200_norm_out_gen22.h5')
model1=load_model('data2/sequentiallstm2222200_b128_h200_norm_out_gen22.h5')



# Calculate predictions


# PredValSet2 = model1.predict(X2.reshape(X2.shape[0],1000,200))
PredValSet2 = model1.predict(X2.reshape(X2.shape[0],200,200))
print("PredValSet2 shape: ",PredValSet2.shape)
print("true_test_output shape: ",true_test_output.shape)


# fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6))=plt.subplots(nrows=3, ncols=2)

# %%
# sample1=np.random.randint(0,100,1)
sample1=np.array([0])
print(sample1[0])

plt.plot(PredValSet2[:,sample1[0]],true_test_output[:,sample1[0]],'o', color='b',markersize=5)
plt.plot(true_test_output[:,sample1[0]],true_test_output[:,sample1[0]],'-', color='r',linewidth=5)
plt.xlabel('prediction',fontsize=22)
plt.ylabel('true value',fontsize=22)
# plt.xticks([-1.00,-0.50,0.00,0.50,1.00])
# plt.yticks([-1.00,-0.50,0.00,0.50,1.00])
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.legend()
plt.show()

# %%
sample1=np.array([100])
print(sample1[0])

plt.plot(PredValSet2[:,sample1[0]],true_test_output[:,sample1[0]],'o', color='b',markersize=5)
plt.plot(true_test_output[:,sample1[0]],true_test_output[:,sample1[0]],'-', color='r',linewidth=5)
plt.xlabel('prediction',fontsize=22)
plt.ylabel('true value',fontsize=22)
# plt.xticks([-1.00,-0.50,0.00,0.50,1.00])
# plt.yticks([-1.00,-0.50,0.00,0.50,1.00])
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.show()
# %%
# plt.plot(PredValSet2[:,sample1[0]],true_test_output[:,sample1[0]],'o', color='b',markersize=5)

# plt.legend()
# plt.show()



# ax1.plot(PredValSet2[:,sample1[0]],true_test_output[:,sample1[0]],'o', color='b',markersize=5)
# ax1.plot(true_test_output[:,sample1[0]],true_test_output[:,sample1[0]],'-', color='r',linewidth=5)

# ax1.legend()




# # sample2=np.random.randint(0,100,1)
# sample2=np.array([82])
# while sample2[0]==sample1[0]:
#     sample2=np.random.randint(0,100,1)

# print(sample2)

# # plt.plot(PredValSet2[:,sample2[0]],true_test_output[:,sample2[0]],'o', color='b',markersize=5)
# # plt.legend()
# # plt.show()


# # plt.xlim(-1, 1)
# # plt.ylim(-1, 1)
# # plt.plot(PredValSet1[:,1],true_test_output[:,1],'o', color='blue',markersize=5,label='lstm')
# ax2.plot(PredValSet2[:,sample2[0]],true_test_output[:,sample2[0]],'o', color='b',markersize=5)
# ax2.plot(true_test_output[:,sample2[0]],true_test_output[:,sample2[0]],'-', color='r',linewidth=5)
# # plt.plot(PredValSet3[:,1],true_test_output[:,1],'o', color='green',markersize=5)
# # plt.plot(PredValSet4[:,1],true_test_output[:,1],'o', color='c',markersize=5)
# # plt.plot(PredValSet5[:,1],true_test_output[:,1],'o', color='m',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# ax2.legend()



# # sample3=np.random.randint(0,100,1)
# sample3=np.array([84])
# while sample3[0]==sample2[0] or sample3[0]==sample1[0]:
#     sample3=np.random.randint(0,100,1)

# print(sample3)

# # # plt.xlim(-1, 1)
# # # plt.ylim(-1, 1)
# # # plt.plot(PredValSet1[:,2],true_test_output[:,2],'o', color='blue',markersize=5,label='lstm')
# # plt.plot(PredValSet2[:,sample3[0]],true_test_output[:,sample3[0]],'o', color='b',markersize=5)
# # # plt.plot(true_test_output[:,2],true_test_output[:,2],'o', color='r',markersize=5)
# # # plt.plot(PredValSet3[:,2],true_test_output[:,2],'o', color='green',markersize=5)
# # # plt.plot(PredValSet4[:,2],true_test_output[:,2],'o', color='c',markersize=5)
# # # plt.plot(PredValSet5[:,2],true_test_output[:,2],'o', color='m',markersize=5)
# # #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# # plt.legend()
# # plt.show()

# # plt.plot(PredValSet1[:,2],true_test_output[:,2],'o', color='blue',markersize=5,label='lstm')
# ax3.plot(PredValSet2[:,sample3[0]],true_test_output[:,sample3[0]],'o', color='b',markersize=5)
# ax3.plot(true_test_output[:,sample3[0]],true_test_output[:,sample3[0]],'-', color='r',linewidth=5)
# # plt.plot(PredValSet3[:,2],true_test_output[:,2],'o', color='green',markersize=5)
# # plt.plot(PredValSet4[:,2],true_test_output[:,2],'o', color='c',markersize=5)
# # plt.plot(PredValSet5[:,2],true_test_output[:,2],'o', color='m',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# ax3.legend()


# # sample4=np.random.randint(0,100,1)
# sample4=np.array([86])
# while sample4[0]==sample2[0] or sample4[0]==sample1[0] or sample4[0]==sample3[0]:
#     sample4=np.random.randint(0,100,1)

# print(sample4)

# # # plt.xlim(-10, 10)
# # # plt.ylim(-10, 10)
# # # plt.plot(PredValSet1[:,3],true_test_output[:,3],'o',color='blue',markersize=5,label='lstm')
# # plt.plot(PredValSet2[:,sample4[0]],true_test_output[:,sample4[0]],'o', color='b',markersize=5)
# # # plt.plot(true_test_output[:,3],true_test_output[:,3],'o', color='r',markersize=5)
# # # plt.plot(PredValSet3[:,3],true_test_output[:,3],'o', color='green',markersize=5)
# # # plt.plot(PredValSet4[:,3],true_test_output[:,3],'o', color='c',markersize=5)
# # # plt.plot(PredValSet5[:,3],true_test_output[:,3],'o', color='m',markersize=5)
# # #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# # plt.legend()
# # plt.show()

# # plt.xlim(-10, 10)
# # plt.ylim(-10, 10)
# # plt.plot(PredValSet1[:,3],true_test_output[:,3],'o',color='blue',markersize=5,label='lstm')
# ax4.plot(PredValSet2[:,sample4[0]],true_test_output[:,sample4[0]],'o', color='b',markersize=5)
# ax4.plot(true_test_output[:,sample4[0]],true_test_output[:,sample4[0]],'-', color='r',linewidth=5)
# # plt.plot(PredValSet3[:,3],true_test_output[:,3],'o', color='green',markersize=5)
# # plt.plot(PredValSet4[:,3],true_test_output[:,3],'o', color='c',markersize=5)
# # plt.plot(PredValSet5[:,3],true_test_output[:,3],'o', color='m',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# ax4.legend()
# # plt.show()

# # sample5=np.random.randint(0,100,1)
# sample5=np.array([88])
# while sample5[0]==sample2[0] or sample5[0]==sample1[0] or sample5[0]==sample3[0] or sample5[0]==sample4[0]:
#     sample5=np.random.randint(0,100,1)

# print(sample5)


# ax5.plot(PredValSet2[:,sample5[0]],true_test_output[:,sample5[0]],'o', color='b',markersize=5)
# ax5.plot(true_test_output[:,sample5[0]],true_test_output[:,sample5[0]],'-', color='r',linewidth=5)

# ax5.legend()




# sample6=np.array([99])
# print(sample6)

# ax6.plot(PredValSet2[:,sample6[0]],true_test_output[:,sample6[0]],'o', color='b',markersize=5)
# ax6.plot(true_test_output[:,sample6[0]],true_test_output[:,sample6[0]],'-', color='r',linewidth=5)
# # plt.plot(PredValSet3[:,3],true_test_output[:,3],'o', color='green',markersize=5)
# # plt.plot(PredValSet4[:,3],true_test_output[:,3],'o', color='c',markersize=5)
# # plt.plot(PredValSet5[:,3],true_test_output[:,3],'o', color='m',markersize=5)
# #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# ax6.legend()


# fig.supxlabel('Prediction')
# fig.supylabel('true value')
# plt.tight_layout()

# plt.show()

# # plt.plot(true_test_output[:,0],color='b',label='r0')
# # plt.plot(true_test_output[:,1],color='r',label='r1')
# # plt.plot(true_test_output[:,2],color='y',label='r2')
# # plt.legend()
# # plt.show()



# ##########################################################################################"

# # predint = model.predict(train_input[:3000])

# # trueint = train_output[:3000]


# # plt.plot(predint[:,3],trueint[:,3],'o', color='blue',markersize=5)
# # #plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
# # plt.show()

