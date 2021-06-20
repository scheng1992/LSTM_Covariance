# coding: utf8
#construction of matrix B and special H with measure on the boarder
import numpy as np
import math
from scipy.linalg import sqrtm
from shallowwater import *

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

#Blgovind
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
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
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
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
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
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B

def bord_M_aleatoire(dimension, proba):
    M=np.zeros((dimension**2,dimension**2))
    for i in range(dimension**2):
        for j in range(dimension**2):
            if j % 10==0 or j % 10==9:
                M[i,j]=np.random.binomial(1, proba)
            elif j<=9 or j>=90:
                M[i,j]=np.random.binomial(1, proba)
    return M       
        
    
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
    
#Blgovind
def Balgovind_noniso(dim,L,rayon): #diag_vect: vector of dim, variance of each point
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=(1+r/L)*(math.exp(-r/L))
            
            rr1 = math.sqrt((a1-5)**2 + (b1-5)**2) #cercle in the middle
            rr2 = math.sqrt((a2-5)**2 + (b2-5)**2)
            if rr1<=rayon: #inside the cercle
                sub_B[i,j]=sub_B[i,j]*4
            else:
                sub_B[i,j]=sub_B[i,j]*0.25
            if rr2<=rayon:
                sub_B[i,j]=sub_B[i,j]*4
            else:
                sub_B[i,j]=sub_B[i,j]*0.25
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B    

if __name__ == "__main__":
    im = plt.imshow(0.01*Balgovind(10,3)[:100,:100])
    #im = plt.imshow(0.01*Balgovind(10,3))
    plt.colorbar(im)
    #plt.savefig("Figures/R_bal_3.eps", format ='eps')