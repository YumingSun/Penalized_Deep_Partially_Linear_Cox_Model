# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:44:40 2022

@author: sunym
"""
import numpy as np
import os
import pandas as pd
import h5py
        
def train_test_read_dplc(dataLoc):
    
    h5f = h5py.File(os.path.join(dataLoc,'select.h5'),'r')
    X = h5f['arr1'][:].T
    h5f.close()
    
    h5f = h5py.File(os.path.join(dataLoc,'no_select.h5'),'r')
    Z = h5f['arr1'][:].T
    h5f.close()

    h5f = h5py.File(os.path.join(dataLoc,'y.h5'),'r')
    y = h5f['arr1'][:].T
    h5f.close()

    _,freq = np.unique(y[:,0],return_counts = True)
    censor_rate = freq[0]/np.sum(freq)
    
    dead_id = (y[:,0] == 1)
    alive_id = (y[:,0] == 0)
    test_alive_size = int(1000 * censor_rate)
    test_dead_size = int(1000 * (1 - censor_rate))
    train_alive_size = int(1500*censor_rate)
    train_dead_size = int(1500*(1-censor_rate))
    
    
    x_dead,z_dead,y_dead = X[dead_id,:], Z[dead_id,:], y[dead_id,:]
    x_alive,z_alive,y_alive = X[alive_id,:], Z[alive_id,:], y[alive_id,:]
    
    x_dead_train, z_dead_train, y_dead_train = \
        x_dead[:train_dead_size,:],z_dead[:train_dead_size,:],y_dead[:train_dead_size,:]
    x_dead_test, z_dead_test, y_dead_test = \
        x_dead[-test_dead_size:,:],z_dead[-test_dead_size:,:],y_dead[-test_dead_size:,:]
    
    x_alive_train, z_alive_train, y_alive_train = \
        x_alive[:train_alive_size,:],z_alive[:train_alive_size,:],y_alive[:train_alive_size,:]
    x_alive_test, z_alive_test, y_alive_test = \
        x_alive[-test_alive_size:,:],z_alive[-test_alive_size:,:],y_alive[-test_alive_size:,:]
        
    x_train = np.concatenate((x_dead_train,x_alive_train),axis = 0)
    z_train = np.concatenate((z_dead_train,z_alive_train),axis = 0)
    y_train = np.concatenate((y_dead_train,y_alive_train),axis = 0)
    
    x_test = np.concatenate((x_dead_test,x_alive_test),axis = 0)
    z_test = np.concatenate((z_dead_test,z_alive_test),axis = 0)
    y_test = np.concatenate((y_dead_test,y_alive_test),axis = 0)
    
    data_x_z_train = pd.DataFrame(np.concatenate((x_train,z_train),axis = 1))
    y_train_order = np.zeros_like(y_train)
    y_train_order[:,0] = y_train[:,1]
    y_train_order[:,1] = y_train[:,0]
    y_train_order = pd.DataFrame(y_train_order)
    
    data_x_z_test = pd.DataFrame(np.concatenate((x_test,z_test),axis = 1))
    y_test_order = np.zeros_like(y_test)
    y_test_order[:,0] = y_test[:,1]
    y_test_order[:,1] = y_test[:,0]
    y_test_order = pd.DataFrame(y_test_order)
    
    return data_x_z_train, y_train_order, data_x_z_test, y_test_order


   