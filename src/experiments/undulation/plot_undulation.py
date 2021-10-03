'''
Created on 3 Oct 2021

@author: lukas
'''


import matplotlib.pyplot as plt
import numpy as np

def plot_head_trajectory(FS):
            
    # Get head position
    X = FS.x[:, :, 0]
    
    # check if motion is planar
    tol = 1e-3    
    X_x = X[:,0]    
    assert np.all(X_x < tol), 'locomotion is not planar'
        
    # plot planar locomotion
    X_y = X[:,1]
    X_z = X[:,2]

    lfz = 16

    fig = plt.figure()
    plt.plot(X_y, X_z)
    plt.xlabel(r'$x$', fontsize = lfz)
    plt.ylabel(r'$y$', fontsize = lfz)

    return fig

