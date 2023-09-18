# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 23:31:52 2022

@author: sunym
"""
import numpy as np
import sys
import pickle
import os
import models_plcox_cd_fit

from utils_plCox import train_test_read_dplc

def experiment(dataLoc,fileId, train_sample_size, linear_dim, neuron1, neuron2,
               dropout1,dropout2):
    non_linear_dim = 8
    data_x, data_y, data_x_test, data_y_test = train_test_read_dplc(
        dataLoc, fileId,train_sample_size)
    
    LAMBDAS = list(np.exp(np.linspace(np.log(5),np.log(0.05),25)))
    lam = [LAMBDAS[20]] # N0500P0600
    # lam = LAMBDAS[20:25] # N1500P0600
    # lam = LAMBDAS[19:23] # N0500P1200
    dplc = models_plcox_cd_fit.DPLC_Non_Linear(linear_dim,non_linear_dim,neuron1,
                                               neuron2,dropout1,dropout2,lam, 
                                               weight_decay=0.01,dnn_epoch = 50,scad_dnn_epoch = 10)
    dplc.fit(data_x,data_y)  
    
    aic_id = dplc._select_best_model('AIC')
    bic_id = dplc._select_best_model('BIC')
    ebic_id = dplc._select_best_model('EBIC')
    
    beta_aic = dplc.BETA[:,aic_id]
    beta_bic = dplc.BETA[:,bic_id]
    beta_ebic = dplc.BETA[:,ebic_id]
    
    false_negative_aic = (beta_aic[:10] == 0).sum()
    false_negative_bic = (beta_bic[:10] == 0).sum()
    false_negative_ebic = (beta_ebic[:10] == 0).sum()
    
    false_positive_aic = (beta_aic[10:] != 0).sum()
    false_positive_bic = (beta_bic[10:] != 0).sum()
    false_positive_ebic = (beta_ebic[10:] != 0).sum()
    
    c_aic_train = dplc.score(data_x,data_y,'AIC')
    c_aic_test = dplc.score(data_x_test,data_y_test,'AIC')
    
    c_bic_train = dplc.score(data_x,data_y,'BIC')
    c_bic_test = dplc.score(data_x_test,data_y_test,'BIC')
    
    c_ebic_train = dplc.score(data_x,data_y,'EBIC')
    c_ebic_test = dplc.score(data_x_test,data_y_test,'EBIC')
    
    nonlinear_estimate_aic_train = dplc.estimate_nonlinear(data_x,'AIC')
    nonlinear_estimate_aic_test = dplc.estimate_nonlinear(data_x_test,'AIC')
    
    nonlinear_estimate_bic_train = dplc.estimate_nonlinear(data_x,'BIC')
    nonlinear_estimate_bic_test = dplc.estimate_nonlinear(data_x_test,'BIC')
    
    nonlinear_estimate_ebic_train = dplc.estimate_nonlinear(data_x,'EBIC')
    nonlinear_estimate_ebic_test = dplc.estimate_nonlinear(data_x_test,'EBIC')
    
    linear_estimate_aic_train = dplc.estimate_linear(data_x,'AIC')
    linear_estimate_aic_test = dplc.estimate_linear(data_x_test,'AIC')
    
    linear_estimate_bic_train = dplc.estimate_linear(data_x,'BIC')
    linear_estimate_bic_test = dplc.estimate_linear(data_x_test,'BIC')
    
    linear_estimate_ebic_train = dplc.estimate_linear(data_x,'EBIC')
    linear_estimate_ebic_test = dplc.estimate_linear(data_x_test,'EBIC')
    
    
    
    
    output = {
        'AIC': {'False Positive': false_positive_aic,
                'False Negative': false_negative_aic, 
                'C Test': c_aic_test, 
                'C Train': c_aic_train,
                'Linear Estimate Test': linear_estimate_aic_test,
                'Linear Estimate Train': linear_estimate_aic_train,
                'Nonlinear Estimate Test': nonlinear_estimate_aic_test,
                'Nonlinear Estimate Train': nonlinear_estimate_aic_train},
        
        'BIC': {'False Positive': false_positive_bic,
                'False Negative': false_negative_bic, 
                'C Test': c_bic_test, 
                'C Train': c_bic_train,
                'Linear Estimate Test': linear_estimate_bic_test,
                'Linear Estimate Train': linear_estimate_bic_train,
                'Nonlinear Estimate Test': nonlinear_estimate_bic_test,
                'Nonlinear Estimate Train': nonlinear_estimate_bic_train},
        
        'EBIC': {'False Positive': false_positive_ebic,
                'False Negative': false_negative_ebic, 
                'C Test': c_ebic_test, 
                'C Train': c_ebic_train,
                'Linear Estimate Test': linear_estimate_ebic_test,
                'Linear Estimate Train': linear_estimate_ebic_train,
                'Nonlinear Estimate Test': nonlinear_estimate_ebic_test,
                'Nonlinear Estimate Train': nonlinear_estimate_ebic_train},
        }
    
    return output,dplc

if __name__ == '__main__':
    neuron1 = int(sys.argv[1])
    neuron2 = int(sys.argv[2])
    dropout1 = float(sys.argv[3])
    dropout2 = float(sys.argv[4])
    train_sample_size = int(sys.argv[5])
    select_dim = int(sys.argv[8])
    
    dataLoc = '../data'
    # dataLoc = '/nfs/turbo/jiankanggroup/yumsun/C90/N{:04d}P{:04d}'.format(train_sample_size,select_dim)
    # resultLoc = '/home/yumsun/DFS/results/experiment/nonlinear/deepPlCox/C{}/N{:04d}P{:04d}/all_res/'.format(signal_strength,
    #                                                                                                 train_sample_size,
    #                                                                                                 select_dim)
    
    
    
    metric,model = experiment(dataLoc,numOfExp,train_sample_size,select_dim,
                              neuron1,neuron2,dropout1,dropout2)
    
    print('False Positive: {}, False Negative: {}, C Test: {}'.format(metric['BIC']['False Positive'], metric['BIC']['False Negative'], metric['BIC']['C Test']))
    pickle.dump(metric, open('results.pkl','wb'))
    pickle.dump(model, open('trained_model.pkl','wb'))
    

