# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 23:31:52 2022

@author: sunym
"""
import sys
import models_plcox_cd_fit

from utils_plCox import train_test_read_dplc

def experiment(dataLoc, neuron1, neuron2, dropout1,dropout):
    
    non_linear_dim = 8
    linear_dim = 600
    data_x, data_y, data_x_test, data_y_test = train_test_read_dplc(
        dataLoc)
    
    dplc = models_plcox_cd_fit.DPLC_Non_Linear(linear_dim,non_linear_dim,neuron1,
                                               neuron2,dropout1,dropout2, 
                                               weight_decay=0.01,dnn_epoch = 50,scad_dnn_epoch = 10)
    dplc.fit(data_x,data_y)  
    
    bic_id = dplc._select_best_model()
    
    beta_bic = dplc.BETA[:,bic_id]
    
    false_negative_bic = (beta_bic[:10] == 0).sum()
    
    false_positive_bic = (beta_bic[10:] != 0).sum()
    
    c_bic_train = dplc.score(data_x,data_y)
    c_bic_test = dplc.score(data_x_test,data_y_test)
    
    
    output = {
                'False Positive': false_positive_bic,
                'False Negative': false_negative_bic, 
                'C Test': c_bic_test, 
                'C Train': c_bic_train
        }
    
    return output,dplc

if __name__ == '__main__':
    neuron1 = int(sys.argv[1])
    neuron2 = int(sys.argv[2])
    dropout1 = float(sys.argv[3])
    dropout2 = float(sys.argv[4])
    
    dataLoc = '../data'
    
    
    
    metric,model = experiment(dataLoc,
                              neuron1,neuron2,dropout1,dropout2)
    
    print('False Positive Number: {}, False Negative Number: {}, C-index: {}'.format(metric['False Positive'], metric['False Negative'], metric['C Test']))
    

