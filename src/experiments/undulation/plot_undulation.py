'''
Created on 3 Oct 2021

@author: lukas
'''


import matplotlib.pyplot as plt
import numpy as np

from cosserat_rod.plot3d import generate_interactive_scatter_clip

def generate_undulation_interactive_video(FS, fps = 500):
        
    generate_interactive_scatter_clip(FS, fps, perspectives = 'yz')
        
    return

def plot_head_trajectory(FS, ax = None):
            
    # Get head position
    X = FS.x[:, :, 0]
    
    # check if motion is planar
    tol = 1e-3    
    X_x = X[:,0]    
    assert np.all(X_x < tol), 'locomotion is not planar'
        
    # plot planar locomotion
    X_y = X[:,1]
    X_z = X[:,2]

    x_y_0 = FS.x[0, 1, :]
    x_z_0 = FS.x[0, 2, :]

    lfz = 16

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    ax.plot(X_y, X_z)
    ax.plot(x_y_0, x_z_0, c = 'r')
    ax.set_xlabel(r'$x$', fontsize = lfz)
    ax.set_ylabel(r'$y$', fontsize = lfz)

    return plt.gcf()

