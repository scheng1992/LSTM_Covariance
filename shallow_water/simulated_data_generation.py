#shallow water propagation 
"""
Solution of Shallow-water equations using a Python class.
Adapted for Python training course at CNRS from https://github.com/mrocklin/ShallowWater/

Dmitry Khvorostyanov, 2015
CNRS/LMD/IPSL, dmitry.khvorostyanov @ lmd.polytechnique.fr
"""

import time
from pylab import *
import matplotlib.gridspec as gridspec
from shallowwater import *
# import imageio
from constructB import *
from DA_preparation import *
import os


if not os.path.isdir('uniform01_2'):
    os.makedirs('uniform01_2')

print("30/05/2021")

class shallow_dynamique(object):
    

    time = 0

    plt = []
    fig = []


    def __init__(self, x=[],y=[],u=zeros((100,100)),v = zeros((100,100)),h=ones((100,100)),dx=0.01,dt=0.0001, N=100,L=1., g=1., b=2.0): # How define no default argument before?


        self.g = g
        self.b = b
        self.L=L
        self.N=N

        # limits for h,u,v
        
        
        self.dx=dx
        self.dt=dt
        
        self.x,self.y = mgrid[:self.N,:self.N]
        
        self.u=u
        self.v=v
        
        self.h=h
        
    # self.h= ones((self.N,self.N))
        
        
        #self.lims = [(self.h_ini-self.Hp,self.h_ini+self.Hp),(-0.02,0.02),(-0.02,0.02)]
        
        

    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (roll(A, -1, axis) - roll(A, 1, axis)) / (self.dx*2.) # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A,1)

    def d_dy(self, A):
        return self.dxy(A,0)


    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]: # type check
            assert isinstance(x, ndarray) and not isinstance(x, matrix)

            g,b,dx = self.g, self.b, self.dx

            du_dt = -g*self.d_dx(h) - b*u
            dv_dt = -g*self.d_dy(h) - b*v

            H = 0 #h.mean() - our definition of h includes this term
            dh_dt = -self.d_dx(u * (h)) - self.d_dy(v * (h))

            return dh_dt, du_dt, dv_dt


    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v

############################################################################

#define H (from x to y in the one dimensional space)

H_1D_uv = np.zeros((100,400))

for i in range(10):
    for j in range(10):
        
        H_1D_uv[i*10+j,2*i*20+2*j] = 1
        
        H_1D_uv[i*10+j,(2*i+1)*20+2*j] = 1
        
        H_1D_uv[i*10+j,2*i*20+2*j+1] = 1
        
        H_1D_uv[i*10+j,(2*i+1)*20+2*j+1] = 1

H_1strow = np.concatenate((H_1D_uv,np.zeros((100,400))),axis = 1)

H_2ndrow = np.concatenate((np.zeros((100,400)),H_1D_uv),axis = 1)


H = np.concatenate((H_1strow,H_2ndrow),axis = 0)
############################################################################

iteration_times=1000


parameter_size=100+1

trainning_set = np.zeros((1,iteration_times*200+parameter_size))

index=0

try:

    for ii in range(0,30000):

        if ii%100==0:
            print(ii)

        #############################################################################

        # define the R matrix

        # D = 0.001*np.random.lognormal(0, 0.7, 100)
        D = 0.001*np.random.uniform(0.01, 0.1, 100)



        r = np.random.uniform(1, 8)



        # same observation error covariance for u and v (D is the lower unit triangle, Balgovind is the diagonal triangle, R is the SPD or HPD)
        R = np.dot(np.dot(np.sqrt(np.diag(D)),Balgovind(10,r)[:100,:100]),np.sqrt(np.diag(D)))

        import copy

        D_copy=D.copy()
        D_copy.shape=(10,10)

        parameters=np.zeros((1,parameter_size))

        parameters[0,:100]=D
        parameters[0,100:]=r




        # v = np.random.uniform(0,100.)

        # R=v*R


        ##############################################################################
        
        SW = shallow(u=np.zeros((20,20)),v=np.zeros((20,20)),px=10,py=10,N = 20,R=10)

        ###############################################################################################
                
        h_ini = SW.h
        
        # SW = shallow_dynamique(u = np.zeros((SW.h.shape[0],SW.h.shape[1])),
        #                     v = np.zeros((SW.h.shape[0],SW.h.shape[1])),
        #                     h = h_ini)

        y_obs = np.zeros((H.shape[0],iteration_times))

        ###############################################################################################

        for i in range(iteration_times):
            print("sample time: ",ii)
            print("iteration time: ",i)
            print(f"index {index}")


            SW.evolve()
            
            Y  = np.dot(H, np.concatenate((SW.u.ravel(),SW.v.ravel())).
                                reshape(2*SW.u.size,1))#generate observations
            
            #R = np.dot(np.dot(np.sqrt(np.diag(D)),np.eye(100)),np.sqrt(np.diag(D)))
            
            
            Y[:100] += np.random.multivariate_normal(np.zeros(100),R).reshape(100,1)# u: update
            
            Y[100:] += np.random.multivariate_normal(np.zeros(100),R).reshape(100,1)# v: update

            # Y: 200*1
            #y_obs: 200*iteration_times

            # ValueError: could not broadcast input array from shape (200,1) into shape (200):
            # https://stackoverflow.com/a/39825046/10349608
            y_obs[:,[i]]=Y

        train_row=y_obs.ravel()
        train_row.shape = (1,train_row.size)

        train_row=np.concatenate((train_row,parameters),axis=1)

        trainning_set=np.concatenate((trainning_set,train_row),axis=0)

        if (ii+1)%500==0:

            index=index+1

            trainning_set = trainning_set[1:,:]

            np.save(f"uniform01_2/trainset_withx_repeat_shwater3_uniform0011_{index}.npy", trainning_set)

            trainning_set = np.zeros((1,iteration_times*200+parameter_size))



        # if (ii+1)%1000==0:

        
            


    



    # np.savetxt("uniform01_2/trainset_withx_repeat_shwater3_uniform0011_total_test6.csv", trainning_set, delimiter=",")

except Exception as e:

    np.savetxt("uniform01_2/trainset_withx_repeat_shwater3_uniform0011_total_test6_1.csv", trainning_set, delimiter=",")

except KeyboardInterrupt:

    np.savetxt("uniform01_2/trainset_withx_repeat_shwater3_uniform0011_total_test6_1.csv", trainning_set, delimiter=",")

except UserAbort:

    np.savetxt("uniform01_2/trainset_withx_repeat_shwater3_uniform0011_total_test6_1.csv", trainning_set, delimiter=",")