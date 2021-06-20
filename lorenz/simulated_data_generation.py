# -*- coding: utf-8 -*-
# generate the trainning set for keras regression 

import numpy as np
from scipy.optimize import fmin
from scipy.optimize import fmin_l_bfgs_b

#from scipy.optimize import fmin_ncg
from scipy.linalg import sqrtm
import math


from constructB import *
from lorentz_attractor import *

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import time
import random
import lorentz_attractor
import sklearn
from sklearn import datasets

import os

if not os.path.isdir('lorenz_cov_train_v2'):
    os.makedirs('lorenz_cov_train_v2')

#######################################################################

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v   #https://www.mygreatlearning.com/blog/covariance-vs-correlation/
    correlation[covariance == 0] = 0
    return correlation

######################################################################
#define matrix R by extra-diagonal elements
def R_covariance_dim3(r1,r2,r3):
    M = np.zeros((3,3))
    M[0,1] = r1
    M[0,2] = r2
    M[1,2] = r3
    M = M + M.T
    M += np.eye(3)
    return M
    
################################################################################
#######################################################################
def cov_to_cor(M): # from a covariance matrix to its associated correlation matrix
    inv_diag_M=np.linalg.inv(sqrtm(np.diag(np.diag(M))))
    cor_M = np.dot(inv_diag_M, np.dot(M,inv_diag_M))
    return cor_M


def lorenz_1step(x, y, z, s=10, r=28, b=2.667,dt = 0.001):
    x_dot, y_dot, z_dot = lorenz(x, y, z)
    x_next = x + (x_dot * dt)
    y_next = y + (y_dot * dt)
    z_next = z + (z_dot * dt)    
    return x_next, y_next, z_next

def VAR_3D(xb,Y,H,B,R): #booleen=1 garde la trace
    xb1=np.copy(xb)
    xb1.shape=(xb1.size,1)
    Y.shape = (Y.size,1)
    dim_x = xb1.size
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.inv(np.dot(H,np.dot(B,np.transpose(H)))+R))) #matrice de gain, Kalman gain
    A=np.dot(np.dot((np.eye(dim_x)-np.dot(K,H)),B),np.transpose((np.eye(dim_x)-np.dot(K,H))))+np.dot(np.dot(K,R),np.transpose(K))  #not the kalman filter expression???
    vect=np.dot(H,xb1)
    xa=np.copy(xb1+np.dot(K,(Y-vect)))

    return xa,A   #xa is the new estimated data, A is the new covariance,
###################################################################################

###################################################################################
    #parameters
num_steps = 1000
H = np.array([[1,1,0],[2,0,1],[0,0,3]])
R = 0.001*np.array([[1,0.4,0.1],[0.4,1,0.4],[0.1,0.4,1]])

B =0.01*np.array([[1,0.2,0.],[0.2,1,0.2],[0.,0.2,1]])
#Q = 0.0001*np.eye(3)

###################################################################################
#save the trainning set for different R 
trainning_set = np.zeros((1,num_steps*3+3+4))   

###################################################################################

#############################################################################
    # true states vector 3 * number_steps
xs,ys,zs = lorenz_attractor(s=10, r=28, b=2.667, dt = 0.001, num_steps=1000)

x_true = np.zeros((3,num_steps+1))

x_true[0,:] = np.copy(xs)
x_true[1,:] = np.copy(ys)
x_true[2,:] = np.copy(zs)


###############################################################################
for ii in range(2000):
    if ii%100 ==0:
        print(ii)
            
# construct observations
                
#=========================================================================
    #generate x with noise
    for repetation in range(10):
        xs,ys,zs = lorenz_attractor(s=10, r=28, b=2.667, dt = 0.001, num_steps = 1000,x0 = 0.+np.random.normal(0, 0.05),
                                    y0=1.+np.random.normal(0, 0.05),z0=1.05+np.random.normal(0, 0.05))
        
        x_true = np.zeros((3,num_steps+1))
        
        x_true[0,:] = np.copy(xs)
        x_true[1,:] = np.copy(ys)
        x_true[2,:] = np.copy(zs)                
        
#=========================================================================

        y_true = np.zeros((3,num_steps+1))
        y_obs = np.zeros((3,num_steps+1))
        
        v = np.random.uniform(0,100.)
        R = correlation_from_covariance(sklearn.datasets.make_spd_matrix(3))  #SPD covariance
        
        r1 = R[0,1]
        r2 = R[0,2]
        r3 = R[1,2]
        
        R = v*R   
        for i in range(num_steps+1):
            print("sample time: ",ii)
            print("iteration time: ",i)
            x = x_true[:,i]
            x.shape = (x.size,1)
            y = np.dot(H,x)               #why this is this expression to calculate y?        
            y.shape = (y.size,)
            y_true[:,i] = y
            
            y_noise = np.random.multivariate_normal(np.zeros(3),R)
            y_noise.shape = (y_noise.size,)
            y_noise += y 
            y_obs[:,i] = y_noise
            
            parameters = np.array([r1,r2,r3,v]) #output for deep learning regression
                        #train_row = np.concatenate((y_obs.ravel(),x_true.ravel())) #input for deep learning    #what are the functionalities of these r ->covaraicen! why v is not necessary???
            train_row = y_obs.ravel()
            train_row = np.concatenate((train_row.ravel(),parameters))
            
        train_row.shape = (1,train_row.size)
        trainning_set = np.concatenate((trainning_set,train_row), axis=0)

        # if repetation+ii*10==5000:

        #     np.savetxt(f"lorenz_cov_train_v2/trainset_withx_steps1000_test_{10000+repetation+ii*10}.csv", trainning_set, delimiter=",")

trainning_set = trainning_set[1:,:]
#####################################################################################""
np.savetxt("lorenz_cov_train_v2/trainset_withx_steps1000_test8.csv", trainning_set, delimiter=",")