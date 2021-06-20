# coding: utf8
#construction of matrix B and special H with measure on the boarder
import numpy as np
import math
from scipy.linalg import sqrtm

##def B_Balgovind(n,Sigma,L):
##    Gamma = np.identity(n)
##    for i in xrange(n):
##        for j in xrange(n):
##            Gamma[i,j] = ( 1. + abs(i-j)/L)*np.exp(-abs(i-j)/L)
##    B = np.dot(Sigma,np.dot(Gamma,Sigma))
##    return B


def get_index_2d (dim,n): #get caratesian coordinate
    j=n % dim
    j=j/1. #float( i)
    i=(n-j)/dim
    return (i,j)# pourquoi float?
#identite
def identiity(n):#n : taille de vecteur xb

    B=np.eye(n)
    return B

#Blgovind (Balgovind correlation functions: file:///C:/paper_ml/sibo/Polyphemus-1.2-Guide-2.pdf)
def Balgovind(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=(1+r/L)*(math.exp(-r/L))
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B [dim**2*2,dim**2*2]
    return B

def Gaussian(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=math.exp(-r**2/(2*L**2))
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B [dim**2*2,dim**2*2]
    return B

def expontielle(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=math.exp(-r/L)
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B [dim**2*2,dim**2*2]
    return B

def bord_M_aleatoire(dimension, proba):
    M=np.zeros((dimension**2,dimension**2))
    for i in range(dimension**2):
        for j in range(dimension**2):
            if j % 10==0 or j % 10==9:
                M[i,j]=np.random.binomial(1, proba)
            elif j<=9 or j>=90:
                M[i,j]=np.random.binomial(1, proba)    #if there is some problems? because both of them are np.random.binomial????
    return M                                           # what are other parameters which are not satisfied with this condition ???
        
    
def cov_to_cor(B):
    inv_diag_B=np.linalg.inv(sqrtm(np.diag(np.diag(B))))
    inv_diag_B=np.copy(inv_diag_B.real)
    cor_B=np.dot(inv_diag_B,np.dot(B,inv_diag_B))
    return cor_B


########################################################################
    #covariance 1d
def Balgovind_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=(1+r/L)*(math.exp(-r/L))

    return B

def expontielle_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=math.exp(-r/L)

    return B

def Gaussian_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=math.exp(-r**2/(2*L**2))

    return B