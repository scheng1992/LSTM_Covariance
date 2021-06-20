# -*- coding: utf-8 -*-
# lorentz system

import numpy as np

import time
import random

import matplotlib.pyplot as plt

import itertools 
import math

from constructB import *

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z;   
           s: sigma
           r: rho
           b: beta
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def lorenz_1step(x, y, z, s=10, r=28, b=2.667,dt = 0.001):
    x_dot, y_dot, z_dot = lorenz(x, y, z)
    x_next = x + (x_dot * dt)
    y_next = y + (y_dot * dt)
    z_next = z + (z_dot * dt)    
    return x_next, y_next, z_next

def lorenz_attractor(s=10, r=28, b=2.667, dt = 0.001,num_steps = 1000,x0=0.,y0=1.,z0=1.05 ):

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    #xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    xs[0] = x0
    ys[0] = y0
    zs[0] = z0
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    return xs,ys,zs

B = 0.01*np.eye(3)
x_noise_initital = np.random.multivariate_normal(np.zeros(3),B)

def lorenz_attractor_noisy(s=10, r=28, b=2.667, dt = 0.001,num_steps = 100000, B = 0.01*np.eye(3), Q = 1e-5*np.eye(3)):

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    #np.random.seed( 10 )
    #x_noise = np.random.multivariate_normal(np.zeros(3),B)
    
    xs[0] += x_noise_initital[0]
    ys[0] += x_noise_initital[1]
    zs[0] += x_noise_initital[2]
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        
        x_noise_step = np.random.multivariate_normal(np.zeros(3),Q)
        
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt) + x_noise_step[0]
        ys[i + 1] = ys[i] + (y_dot * dt) + x_noise_step[1]
        zs[i + 1] = zs[i] + (z_dot * dt) + x_noise_step[2]

    return xs,ys,zs
# Plot
if __name__ == '__main__' :
        
# plot for amusing
#    for i in range(1200,1400):
#        xs,ys,zs = lorenz_attractor(num_steps = 10*i)
#        xb,yb,zb = lorenz_attractor(num_steps = 10*i-300)
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        
#        #ax.plot(xs, ys, zs, lw=0.5)
#        ax.scatter(xs, ys, zs, lw=0.5)
#        ax.scatter(xb, yb, zb, 'r',lw=0.35)
#        ax.set_xlabel("X Axis")
#        ax.set_ylabel("Y Axis")
#        ax.set_zlabel("Z Axis")
#        ax.set_title("Lorenz Attractor")
#        plt.savefig('tmp_figure/lorenz_catch_'+str(i)+'.png', format='png')
#        #plt.show()
#        #plt.pause(3)
#        plt.close()
#        
#    import imageio
#    images = []
#    for i in range(1200,1400):
#        images.append(imageio.imread('tmp_figure/lorenz_catch_'+str(i)+'.png'))
#    imageio.mimsave('tmp_figure/lorenz_catch.gif', images)
##    
    
#    import glob
#    from PIL import *
#    # Create the frames
#    frames = []
#    imgs = glob.glob("*.png")
#    for i in range(2000):
#        new_frame = Image.open('tmp_figure/lorenz_compose_'+str(i)+'.png')
#        frames.append(new_frame)
#     
#    # Save into a GIF file that loops forever
#    frames[0].save('tmp_figure/png_to_gif.gif', format='GIF',
#                   append_images=frames[1:],
#                   save_all=True,
#                   duration=300, loop=0)
    
    
#############################################################################################"
    # print noisy system
    xs,ys,zs = lorenz_attractor_noisy(num_steps = 1400)
# plot for amusing
    for i in range(1200,1400):
        #xs,ys,zs = lorenz_attractor_noisy(num_steps = 10*i)
        #xb,yb,zb = lorenz_attractor(num_steps = 10*i-300)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #ax.plot(xs, ys, zs, lw=0.5)
        ax.scatter(xs[:i], ys[:i], zs[:i], lw=0.5)
        #ax.scatter(xb, yb, zb, 'r',lw=0.35)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.savefig('tmp_figure/lorenz_noisy_'+str(i)+'.png', format='png')
        #plt.show()
        #plt.pause(3)
        plt.close() 
        
    import imageio
    images = []
    for i in range(1200,1400):
        images.append(imageio.imread('tmp_figure/lorenz_noisy_'+str(i)+'.png'))
    imageio.mimsave('tmp_figure/lorenz_noisy.gif', images)