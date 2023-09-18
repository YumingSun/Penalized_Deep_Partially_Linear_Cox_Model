# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:44:40 2022

@author: sunym
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
import pickle
import os
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import h5py

def survival_linear_simulator(p_select,p_no_select,N,s,censor_rate,lam,
                              beta_lower,beta_higher,alpha_lower, alpha_higher, 
                              outputAd = None, fileId = None):
    p_all = p_select + p_no_select
    beta = np.zeros((p_select,1))
    non_zero = np.arange(s)
    beta[non_zero,0] = np.random.uniform(beta_lower,beta_higher, s)
    
    alpha = np.random.uniform(alpha_lower,alpha_higher, (p_no_select,1))
    
    avg = np.zeros((p_all,))
    cov = np.identity(p_all)
    upper_tri_ind = np.triu_indices(p_all,1)
    lower_tri_ind = np.tril_indices(p_all,-1)
    cov[upper_tri_ind] = 0.2
    cov[lower_tri_ind] = 0.2

    mvn_generator = multivariate_normal(mean = avg,cov = cov)

    mvn_sample = mvn_generator.rvs(size = N)

    X_scale = mvn_sample[:,:p_select]
    Z_scale = mvn_sample[:,p_select:]

    riskSelect = np.dot(X_scale,beta)
    riskNoSelect = np.dot(Z_scale,alpha) 
    # np.mean(riskNoSelect)    
    
    risk_scores = riskSelect + riskNoSelect
    
    unif = np.random.uniform(size = (N,1))
    
    time = -np.log(unif)/(lam*np.exp(risk_scores))
    
    censor_time = np.random.uniform(0,6500,size = (N,1))
    # n, bins, patches = plt.hist(time,60, density=True)


    survCensor = np.ones((N,2))
    censorId = np.where(censor_time < time)[0]
    # censorId = np.random.choice(range(N),size = (int(N * censor_rate),),replace=False)
    survCensor[censorId,0] = 0
    survCensor[:,1] = np.squeeze(time)
    survCensor[censorId,1] = censor_time[censorId,0]
    
    # _,freq = np.unique(survCensor[:,0],return_counts = True)
    # censor_rate = freq[0]/np.sum(freq)
   
    

    survCensor = np.core.records.fromarrays(survCensor.transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')

    cstat = concordance_index_censored(survCensor["Status"], survCensor["Survival_in_days"], 
                               np.squeeze(risk_scores))[0]
    # print("C_stat: {}, censor: {}, mean_g: {}".format(cstat, censor_rate,  np.mean(riskNoSelect)))
    
    

    survCensor_numeric = survCensor.astype(np.dtype([('Status', 'i8'), ('Survival_in_days', 'f8')]))
    survCensor_numeric = np.array(survCensor_numeric.tolist())

    if outputAd is None:
        return X_scale, Z_scale,survCensor_numeric,beta,cstat,alpha,riskNoSelect
    else:
        np.savetxt(os.path.join(outputAd,'x',
                                'select_{:02d}.csv'.format(fileId)),
                   X_scale,delimiter=',')

        np.savetxt(os.path.join(outputAd,'x',
                                'no_select_{:02d}.csv'.format(fileId)),
                   Z_scale,delimiter=',')

        np.savetxt(os.path.join(outputAd,'y',
                                'y_{:02d}.csv'.format(fileId)),
                   survCensor_numeric,delimiter=',')

        np.savetxt(os.path.join(outputAd,'coef',
                                'coef_{:02d}.csv'.format(fileId)),
                   beta,delimiter=',')

        np.savetxt(os.path.join(outputAd,'cStat',
                                'cStat_{:02d}.csv'.format(fileId)),
                   np.array(cstat).reshape((1,1)),delimiter=',')

        
# p_no_select = 8
# p_select = 500
# N = 3000
# s = 10
# censor_rate = 0.3
# beta_lower = -2
# beta_higher = 2
# normalSd = 1
# scaleData = True
# lam = 1/365
# tuneC = 0.9

def survival_nonlinear_simulator(p_select,p_no_select,N,s,censor_rate,lam,
                              beta_lower,beta_higher, tuneC,
                              normal_scale,scale_data,
                              outputAd = None, fileId = None):
    
    beta = np.zeros((p_select,1))
    non_zero = np.arange(s)
    beta[non_zero,0] = np.random.uniform(beta_lower,beta_higher, s)
    
    p_all = p_select + p_no_select
    
    avg = np.zeros((p_all,))
    cov = np.identity(p_all)
    upper_tri_ind = np.triu_indices(p_all,1)
    lower_tri_ind = np.tril_indices(p_all,-1)
    cov[upper_tri_ind] = 0.2
    cov[lower_tri_ind] = 0.2
    
    mvn_generator = multivariate_normal(mean = avg,cov = cov)
    
    mvn_sample = mvn_generator.rvs(size = N)
    
    X_scale = mvn_sample[:,:p_select]
    Z_scale = mvn_sample[:,p_select:]
    
    riskSelect = np.dot(X_scale,beta)
    # plt.boxplot(riskSelect)
    # plt.show()
    riskNoSelect = 0.75*np.exp(Z_scale[:, 0]) -\
        0.5 * np.log((Z_scale[:, 1] - Z_scale[:,2])**2) + \
        0.35*np.sin(Z_scale[:, 3]*Z_scale[:, 4]) - \
            0.5*(Z_scale[:, 5] - Z_scale[:,6] + Z_scale[:,7])**2
    riskNoSelect = riskNoSelect * tuneC - 0.32
    # plt.boxplot(riskNoSelect)
    # plt.show()
    risk_scores = riskSelect + np.expand_dims(riskNoSelect, 1)
    
    unif = np.random.uniform(size = (N,1))
    
    time = -np.log(unif)/(lam*np.exp(risk_scores))
    
    censor_time = np.random.uniform(0,4500,size = (N,1))
    # n, bins, patches = plt.hist(time,60, density=True)
    # plt.show()

    survCensor = np.ones((N,2))
    censorId = np.where(censor_time < time)[0]
    # censorId = np.random.choice(range(N),size = (int(N * censor_rate),),replace=False)
    survCensor[censorId,0] = 0
    survCensor[:,1] = np.squeeze(time)
    survCensor[censorId,1] = censor_time[censorId,0]

    survCensor = np.core.records.fromarrays(survCensor.transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')

    cstat = concordance_index_censored(survCensor["Status"], survCensor["Survival_in_days"], 
                               np.squeeze(risk_scores))[0]
    
    

    survCensor_numeric = survCensor.astype(np.dtype([('Status', 'i8'), ('Survival_in_days', 'f8')]))
    survCensor_numeric = np.array(survCensor_numeric.tolist())

    if outputAd is None:
        return X_scale, Z_scale,survCensor_numeric,beta,cstat,riskNoSelect
    else:
        np.savetxt(os.path.join(outputAd,'x',
                                'select_{:02d}.csv'.format(fileId)),
                   X_scale,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'x',
                                'no_select_{:02d}.csv'.format(fileId)),
                   Z_scale,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'y',
                                'y_{:02d}.csv'.format(fileId)),
                   survCensor_numeric,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'coef',
                                'coef_{:02d}.csv'.format(fileId)),
                   beta,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'cStat',
                                'cStat_{:02d}.csv'.format(fileId)),
                   np.array(cstat).reshape((1,1)),delimiter=',')
        

def survival_nonlinear_simulator_mis(p_select,p_no_select,N,s,censor_rate,lam, 
                                     tuneC, 
                                     normal_scale,scale_data, 
                                     outputAd = None, fileId = None):
    
    
    X_orginal = np.random.normal(0,normal_scale, (N, p_select))
    rho = 0.2
    X = np.zeros((N, p_select))
    X[:, 0] = X_orginal[:, 0]
    X[:, p_select-1] = X_orginal[:, p_select-1]
    for i in range(1, p_select-1):
        X[:, i] = X_orginal[:, i] + rho * (X_orginal[:, i-1] + X_orginal[:, i+1])
    
    Z_orginal = np.random.normal(0,normal_scale, (N, p_no_select))
    rho = 0.2
    Z = np.zeros((N, p_no_select))
    Z[:, 0] = Z_orginal[:, 0]
    Z[:, p_no_select-1] = X_orginal[:, p_no_select-1]
    for i in range(1, p_no_select-1):
        Z[:, i] = Z_orginal[:, i] + rho * (Z_orginal[:, i-1] + Z_orginal[:, i+1])
    
    if scale_data:
        X_scale = (X - X.mean(axis = 0))/X.std(axis = 0)
        Z_scale = (Z - Z.mean(axis = 0))/Z.std(axis = 0)
        
        riskSelect = 0.75*np.exp(X_scale[:, 0]) -\
            0.5 * np.log((X_scale[:, 1] - X_scale[:,2])**2) + \
            0.35*np.sin(X_scale[:, 3]*X_scale[:, 4]) - \
                0.5*(X_scale[:, 5] - X_scale[:,6] + X_scale[:,7])**2 + \
                    0.75 * (X_scale[:, 8] + X_scale[:, 9])**2
        
        # plt.boxplot(riskSelect)
        # plt.show()
        riskNoSelect = 0.75*np.exp(Z_scale[:, 0]) -\
            0.5 * np.log((Z_scale[:, 1] - Z_scale[:,2])**2) + \
            0.35*np.sin(Z_scale[:, 3]*Z_scale[:, 4]) - \
                0.5*(Z_scale[:, 5] - Z_scale[:,6] + Z_scale[:,7])**2
        riskNoSelect = riskNoSelect * tuneC
        # plt.boxplot(riskNoSelect)
        # plt.show()
        risk_scores = np.expand_dims(riskSelect, 1) + np.expand_dims(riskNoSelect, 1)
    else:
        X_scale = X
        Z_scale = Z
        riskSelect = 0.75*np.exp(X_scale[:, 0]) -\
            0.5 * np.log((X_scale[:, 1] - X_scale[:,2])**2) + \
            0.35*np.sin(X_scale[:, 3]*X_scale[:, 4]) - \
                0.5*(X_scale[:, 5] - X_scale[:,6] + X_scale[:,7])**2 + X_scale[:, 8 ] + \
                    X_scale[:, 9 ]
        # plt.boxplot(riskSelect)
        riskNoSelect = 0.75*np.exp(Z_scale[:, 0]) -\
            0.5 * np.log((Z_scale[:, 1] - Z_scale[:,2])**2) + \
            0.35*np.sin(Z_scale[:, 3]*Z_scale[:, 4]) - \
                0.5*(Z_scale[:, 5] - Z_scale[:,6] + Z_scale[:,7])**2
        riskNoSelect = riskNoSelect * tuneC
        # plt.boxplot(riskNoSelect)
        
        risk_scores = riskSelect + np.expand_dims(riskNoSelect, 1)
    
    
    unif = np.random.uniform(size = (N,1))
    
    time = -np.log(unif)/(lam*np.exp(risk_scores))

    n, bins, patches = plt.hist(time,60, density=True)
    plt.show()

    survCensor = np.ones((N,2))
    censorId = np.random.choice(range(N),size = (int(N * censor_rate),),replace=False)
    survCensor[censorId,0] = 0
    survCensor[:,1] = np.squeeze(time)
    survCensor[censorId,1] = np.random.uniform(high = time[censorId]).squeeze()

    survCensor = np.core.records.fromarrays(survCensor.transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')

    cstat = concordance_index_censored(survCensor["Status"], survCensor["Survival_in_days"], 
                               np.squeeze(risk_scores))[0]
    
    

    survCensor_numeric = survCensor.astype(np.dtype([('Status', 'i8'), ('Survival_in_days', 'f8')]))
    survCensor_numeric = np.array(survCensor_numeric.tolist())

    if outputAd is None:
        return X_scale, Z_scale,survCensor_numeric,cstat
    else:
        np.savetxt(os.path.join(outputAd,'x',
                                'select_{:02d}.csv'.format(fileId)),
                   X_scale,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'x',
                                'no_select_{:02d}.csv'.format(fileId)),
                   Z_scale,delimiter=',')
        
        np.savetxt(os.path.join(outputAd,'y',
                                'y_{:02d}.csv'.format(fileId)),
                   survCensor_numeric,delimiter=',')
        
        
        np.savetxt(os.path.join(outputAd,'cStat',
                                'cStat_{:02d}.csv'.format(fileId)),
                   np.array(cstat).reshape((1,1)),delimiter=',')
        
        
def train_test_read_dplc(dataLoc, fileId,train_sample_size):
    
    h5f = h5py.File(os.path.join(dataLoc,
        'select_{:03d}.h5'.format(fileId)),'r')
    X = h5f['arr1'][:].T
    h5f.close()
    
    h5f = h5py.File(os.path.join(dataLoc,
        'no_select_{:03d}.h5'.format(fileId)),'r')
    Z = h5f['arr1'][:].T
    h5f.close()

    h5f = h5py.File(os.path.join(dataLoc,
        'y_{:03d}.h5'.format(fileId)),'r')
    y = h5f['arr1'][:].T
    h5f.close()

    # X = np.genfromtxt(os.path.join(dataLoc,'x',
    #                                'select_{:03d}.csv'.format(fileId)),
    #                   delimiter = ',')
    
    # Z = np.genfromtxt(os.path.join(dataLoc,'x',
    #                                'no_select_{:03d}.csv'.format(fileId)),
    #                   delimiter = ',')
    
    # y = np.genfromtxt(os.path.join(dataLoc,'y',
    #                                'y_{:03d}.csv'.format(fileId)),
    #                  delimiter = ',')
    
    _,freq = np.unique(y[:,0],return_counts = True)
    censor_rate = freq[0]/np.sum(freq)
    
    # beta = np.genfromtxt(os.path.join(dataLoc,'coef',
    #                                'coef_{:02d}.csv'.format(fileId)),
    #                   delimiter = ',')
    # s = (beta != 0).sum()
    
    dead_id = (y[:,0] == 1)
    alive_id = (y[:,0] == 0)
    test_alive_size = int(1000 * censor_rate)
    test_dead_size = int(1000 * (1 - censor_rate))
    train_alive_size = int(train_sample_size*censor_rate)
    train_dead_size = int(train_sample_size*(1-censor_rate))
    
    
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

def stratified_cv(data,trainSize = 0.8, nFold = 5):
    '''

    Parameters
    ----------
    data : dataframe
        the first column is event indicator
    train_size : float
        The default is 0.8.
    nFold : float
        The default is 5.

    Returns
    -------
    An iterable of length n_folds, each element of which is a 2-tuple of numpy
    1-d arrays (train_index, test_index) containing the indices of the test and
    training sets for that cross-validation run

    '''
    id1 = np.where(data.iloc[:,1] == 1)[0]
    id0 = np.where(data.iloc[:,1] == 0)[0]
    myCvIterator = []
    for i in range(nFold):
        id1_train,id1_test = train_test_split(id1, train_size=trainSize)
        id0_train,id0_test = train_test_split(id0, train_size=trainSize)
        idTrain = np.concatenate((id1_train,id0_train))
        np.random.shuffle(idTrain)
        idTest = np.concatenate((id1_test,id0_test))
        np.random.shuffle(idTest)
        myCvIterator.append( (idTrain,idTest) )
    return myCvIterator

if __name__ == '__main__':
    p_no_select = 8
    p_select = 500
    N = 3000
    s = 10
    censor_rate = 0.3
    beta_l = -2
    beta_u = 2
    normalSd = 1
    scaleData = True
    lam = 1/365

    cstat = 0
    mean_g = 99
    
    while ((cstat > 0.935) or (cstat < 0.925)) or (mean_g >= 0.01):
        X_scale, Z_scale, survCensor_numeric,beta,cstat,g = \
            survival_nonlinear_simulator(p_select,p_no_select,N,s,censor_rate,
                                         lam,beta_l,beta_u,
                                         0.9,
                                         normalSd, scaleData)
        mean_g = np.abs(np.mean(g))
        print(cstat)
        print(np.mean(g))
    
    np.mean(g)
    np.quantile(g,[0.25,0.5,0.75])
    
    
    import numpy as np
    from scipy.stats import multivariate_normal
    
    p = 10
    avg = np.zeros((p,))
    cov = np.identity(p)
    upper_tri_ind = np.triu_indices(p,1)
    lower_tri_ind = np.tril_indices(p,-1)
    cov[upper_tri_ind] = 0.2
    cov[lower_tri_ind] = 0.2
    
    mvn_generator = multivariate_normal(mean = avg,cov = cov)
    
    mvn_sample = mvn_generator.rvs(size = 1000)
    
    np.corrcoef(mvn_sample,rowvar = False)
