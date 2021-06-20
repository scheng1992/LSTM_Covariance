"""
Solution of Shallow-water equations using a Python class.
Adapted for Python training course at CNRS from https://github.com/mrocklin/ShallowWater/

Dmitry Khvorostyanov, 2015
CNRS/LMD/IPSL, dmitry.khvorostyanov @ lmd.polytechnique.fr
"""

import time
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
#construct background states, observations with error

def x_to_y(X): # averaging in 2*2 windows (4 pixels)
    dim = X.shape[0]
    dim = 20
    Y = np.zeros((dim/2,dim/2))
    for i in range(dim/2):
        for j in range(dim/2):
            Y[i,j] = X[2*i,2*j] + X[2*i+1,2*j] + X[2*i,2*j+1] + X[2*i+1,2*j+1]
            
            Y_noise = np.random.multivariate_normal(np.zeros(100),0.0000 * np.eye(100))
            Y_noise.shape = (10,10)
            Y = Y + Y_noise
    return Y
    

class shallow(object):

    # domain

    #N = 100
    #L = 1.
    #dx =  L / N
    #dt = dx / 100.

    # Initial Conditions

    #u = zeros((N,N)) # velocity in x direction
    #v = zeros((N,N)) # velocity in y direction

    #h_ini = 1.
    #h = h_ini * ones((N,N)) # pressure deviation (like height)
    #x,y = mgrid[:N,:N]

    time = 0

    plt = []
    fig = []


    def __init__(self, x=[],y=[],h_ini = 1.,u=[],v = [],dx=0.01,dt=0.0001, N=100,L=1., px=50, py=50, R=100, Hp=0.1, g=1., b=2.): # How define no default argument before?


        # add a perturbation in pressure surface
        

        self.px, self.py = px, py
        self.R = R
        self.Hp = Hp

        

        # Physical parameters

        self.g = g
        self.b = b
        self.L=L
        self.N=N

        # limits for h,u,v
        
        
        #self.dx =  self.L / self.N # a changer
        #self.dt = self.dx / 100.
        self.dx=dx
        self.dt=dt
        
        self.x,self.y = mgrid[:self.N,:self.N]
        
        self.u=zeros((self.N,self.N))
        self.v=zeros((self.N,self.N))
        
        self.h_ini=h_ini
        
        self.h=self.h_ini * ones((self.N,self.N))
        
        rr = (self.x-px)**2 + (self.y-py)**2
        self.h[rr<R] = self.h_ini + Hp #set initial conditions
        
        self.lims = [(self.h_ini-self.Hp,self.h_ini+self.Hp),(-0.02,0.02),(-0.02,0.02)]
        
        

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
        dh_dt = -self.d_dx(u * (H+h)) - self.d_dy(v * (H+h))

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


##    def plot(self,tit=None,autolims=False):
##        """Plot u,v,h at current state."""
##
##        self.fig.append(figure())
##        for i,v in enumerate([self.h,self.u,self.v]):
##            if autolims:
##                vmin,vmax = None, None
##            else:
##                vmin, vmax = self.lims[i][0], self.lims[i][1]
##
##            self.fig[-1].add_subplot(3,1,i+1)
##
##            self.plt.append(pcolormesh(v, vmin=vmin, vmax=vmax))
##            colorbar(shrink=0.9)
##
##            if i==0:
##                if tit is None:
##                    self.tit = title('At time %f'% self.time)
##                else:
##                    self.tit = title(tit)
##
##
##    def animate(self):
##        """Plot u,v,h at current state."""
##
##        for i,v in enumerate([self.h,self.u,self.v]):
##            self.plt[i].set_array(v.ravel())
##            
##            if i==0:
##                self.tit.set_text('At time %f'%self.time)



if __name__ == '__main__': #run the current script

    iteration_times= 500
    SW = shallow(N=20,px=10,py=10,R=10.)
    # chose a point (x,y) to check the evolution
    x=10
    y=10

    #SW.plot()
#    u_vect=np.zeros(iteration_times)
#    v_vect=np.zeros(iteration_times)
#    h_vect=np.zeros(iteration_times)
#    for i in range(iteration_times):
#        SW.evolve()
#        u_vect[i]=SW.u[x][y]
#        v_vect[i]=SW.v[x][y]
#        h_vect[i]=SW.h[x][y]
#        #SW.animate()
#
#        if i % 100 == 0:
#            print ('time %f'%SW.time)
##            SW.fig[-1].savefig('sw_%.3d.png'% i)
##
##    show()

#    print SW.time
##    plt.subplot(311)
##    plt.imshow( SW.h)
##    set_title("Title for first plot")
##
##
##    plt.subplot(312)
##    plt.imshow( SW.u)
##   
##
##
##    plt.subplot(313)
##    plt.imshow( SW.v)
##    
##    plt.show()



    gs = gridspec.GridSpec(2, 2,
                    width_ratios=[1, 1],
                    height_ratios=[1, 1]
                    )
    fig = plt.figure()
    
    #fig = plt.figure()
    t=SW.time


  

    fig = plt.figure()
    t=SW.time
    fig.suptitle('At time: T=%1.3f'%t, fontsize=16)
    ax1 = plt.subplot(gs[0])
    ax1.set_title("h")
    im=ax1.imshow( SW.h)
    plt.colorbar(im)

    

    ax2 = plt.subplot(gs[1])
    ax2.set_title("u")
    im=ax2.imshow( SW.u)
    plt.colorbar(im)


    ax3 = plt.subplot(gs[2])
    ax3.set_title("v")
    im=ax3.imshow( SW.v)
    plt.colorbar(im)


    plt.show()

    gs = gridspec.GridSpec(1, 2,
                    width_ratios=[1, 1],
                    )
    fig = plt.figure()
    
    #fig = plt.figure()
    t=SW.time


  ################################################################################
    #u_t=SW.u[[45,46,47,48,49,51,52,53,54,55], :][:, [45,46,47,48,49,51,52,53,54,55]]
    #v_t=SW.v[[45,46,47,48,49,51,52,53,54,55], :][:, [45,46,47,48,49,51,52,53,54,55]]

    iteration_times = 5000
    
    for i in range(iteration_times):
        SW.evolve()
        
        if i%100 ==0:
            u_t = SW.u
            v_t = SW.v
            
#            fig = plt.figure()
#            t=SW.time
#            fig.suptitle('At time: T=%1.3f'%t, fontsize=16)
#            ax1 = plt.subplot(gs[0])
#            ax1.set_title("u")
#            im=ax1.imshow( u_t,interpolation='none')
#            plt.colorbar(im)
#        
#            ax2 = plt.subplot(gs[1])
#            ax2.set_title("v")
#            im=ax2.imshow( v_t,interpolation='none')
#            plt.colorbar(im)
#        
#            plt.show()
            
            
            im = plt.imshow(SW.h, interpolation = "None")
            plt.colorbar(im)
            plt.title("$h (t = 0.05)$")
            #plt.savefig("Figures/h_t005.eps", format = "eps")
            plt.show()
            plt.close()
        
            im = plt.imshow(SW.u, interpolation = "None")
            plt.colorbar(im)
            plt.title("$u (t = 0.05)$")
            #plt.savefig("Figures/u_t005.eps", format = "eps")
            plt.show()
            plt.close()        
                
            im = plt.imshow(SW.v, interpolation = "None")
            plt.colorbar(im)
            plt.title("$v (t = 0.05)$")
            #plt.savefig("Figures/v_t005.eps", format = "eps")
            plt.show()
            plt.close()   

#    i = 300
#    xbu = np.loadtxt("data/SW_b_u_"+str(i)+".txt") 
#    xbv = np.loadtxt("data/SW_b_v_"+str(i)+".txt")     
#    xbh = np.loadtxt("data/SW_b_h_"+str(i)+".txt")     
#
#    im = plt.imshow(xbu, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$u_b (t = 0.05)$")
#    #plt.savefig("Figures/u_b005.eps", format = "eps")
#    plt.show()
#    plt.close() 
#    
#    im = plt.imshow(xbv, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$v_b (t = 0.05)$")
#    #plt.savefig("Figures/v_b005.eps", format = "eps")
#    plt.show()
#    plt.close() 
#    
#    im = plt.imshow(xbh, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$h_b (t = 0.05)$")
#    #plt.savefig("Figures/h_b005.eps", format = "eps")
#    plt.show()
#    plt.close()     
#    
#    yu = x_to_y(SW.u)
#    yv = x_to_y(SW.v)
#    
#    im = plt.imshow(yu, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$y_{t,u} (t = 0.05)$")
#    #plt.savefig("Figures/yt_u005.eps", format = "eps")
#    plt.show()
#    plt.close()       
#    
#    im = plt.imshow(yv, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$y_{t,v} (t = 0.05)$")
#    #plt.savefig("Figures/yt_v005.eps", format = "eps")
#    plt.show()
#    plt.close() 
#    
#    ybu = np.loadtxt("data/SW_yu_"+str(i)+".txt")
#    ybv = np.loadtxt("data/SW_yv_"+str(i)+".txt")
#    
#    im = plt.imshow(ybu, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$y_{b,u} (t = 0.05)$")
#    #plt.savefig("Figures/yb_u005.eps", format = "eps")
#    plt.show()
#    plt.close()       
#    
#    im = plt.imshow(ybv, interpolation = "None")
#    plt.colorbar(im)
#    plt.title("$y_{b,v} (t = 0.05)$")
#    #plt.savefig("Figures/yb_v005.eps", format = "eps")
#    plt.show()
#    plt.close() 