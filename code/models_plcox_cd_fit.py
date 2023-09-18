# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:07:11 2022

@author: sunym
"""
import torch
import torch.nn.functional as F
import numpy as np
from sksurv.metrics import concordance_index_censored
from DPLC_linear_fit import deep_pl_scad_fit, standardize
# from DPLC_linear_fit_simple import deep_pl_scad_fit, standardize
from DPLC_nonlinear_fit import deep_pl_dnn_fit
from sksurv.linear_model import CoxnetSurvivalAnalysis
from partialLik_plcox import coxph_loss_scad_like

class Net_nonlinear(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, 
                 dropout_rate1, dropout_rate2, n_output):
        super(Net_nonlinear, self).__init__() 
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1).double() 
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2).double()
        self.dropout1 = torch.nn.Dropout(p=dropout_rate1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate2)
        self.out = torch.nn.Linear(n_hidden2, n_output).double()
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x) 
        return x
    
class Net_linear(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Net_linear, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = torch.nn.Linear(input_dim, output_dim,bias = False).double()
    
    def forward(self,x):
        return self.hidden(x)


class DPLC_Linear:
    def __init__(self, linear_dim, non_linear_dim,
                 scad_LAMBDA,scad_a = 3.7,
                 learning_rate = 0.01, weight_decay = 0.1,
                 dnn_epoch = 100, scad_dnn_epoch = 10):
        
        self.linear_dim = linear_dim
        self.non_linear_dim = non_linear_dim
        self.learning_rate = learning_rate
        self.scad_a = scad_a
        self.scad_LAMBDA = scad_LAMBDA
        self.weight_decay = weight_decay
        self.dnn_epoch = dnn_epoch
        self.scad_dnn_epoch = scad_dnn_epoch
        
    def get_params(self,deep=True):
        return {"linear_dim" : self.linear_dim,
                "non_linear_dim" : self.non_linear_dim,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scad_a": self.scad_a,
                "scad_lam": self.scad_LAMBDA,
                "dnn_epoch": self.dnn_epoch,
                "scad_dnn_epoch": self.scad_dnn_epoch}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,data_x,data_y):
        self.sample_size = data_x.shape[0]
        t_ord = np.argsort(data_y.iloc[:,0])
        order_data_x = data_x.iloc[t_ord,:]
        order_data_y = torch.tensor(data_y.iloc[t_ord,:].to_numpy(),
                                           dtype = torch.double)
        
        #prepare data for linear part 
        order_linear_data_x = torch.tensor(order_data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_order_linear_data_list = standardize(order_linear_data_x)
        std_order_linear_data_x = std_order_linear_data_list['X']
        self.center = std_order_linear_data_list['center']
        self.scale = std_order_linear_data_list['scale']
        
        #prepare data for nonlinear part
        order_nonlinear_data_x = torch.tensor(order_data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_order_nonlinear_data_list = standardize(order_nonlinear_data_x)
        std_order_nonlinear_data_x = std_order_nonlinear_data_list['X']
        
        
        #set up model
        lambda_len = len(self.scad_LAMBDA)
        self.LOSS = np.zeros((lambda_len,),dtype = np.float64)
        self.BETA = np.zeros((self.linear_dim,lambda_len),dtype = np.float64)
        self.MODEL = []
        lam_id = 0

        for l in self.scad_LAMBDA:
            loss = []
            beta = torch.zeros(self.linear_dim,1,dtype = torch.double)
            dnn  = Net_linear(self.non_linear_dim,1) 
            
            best_dnn = Net_linear(self.non_linear_dim,1) 
            best_beta = torch.zeros_like(beta)
            for i in range(self.scad_dnn_epoch):
                ## fit nonlinear DNN model
                linear_predict = torch.matmul(std_order_linear_data_x, beta)
                deep_pl_dnn_fit(std_order_nonlinear_data_x,
                                linear_predict,
                                order_data_y, 
                                dnn, 
                                self.weight_decay,
                                self.learning_rate,
                                self.dnn_epoch)
                
                ## fit linear model
                dnn.eval()
                nonlinear_predict = dnn.forward(std_order_nonlinear_data_x).data
                dnn.train()
                beta = deep_pl_scad_fit(std_order_linear_data_x, 
                                        order_data_y[:,1], 
                                        beta, nonlinear_predict, 
                                        l, self.linear_dim)
                linear_predict = torch.matmul(std_order_linear_data_x, beta)
                loss.append(coxph_loss_scad_like(nonlinear_predict + linear_predict,
                                 order_data_y))
                if loss[-1] == min(loss):
                    best_dnn.load_state_dict(dnn.state_dict())
                    best_beta = beta
                    

            self.LOSS[lam_id] = min(loss).item()
            self.BETA[:,lam_id] = best_beta.numpy().squeeze()
            self.MODEL.append(best_dnn)
            print(best_beta.numpy())
            lam_id = lam_id + 1
            print(lam_id)
            
        return self
    
    def _select_best_model(self,criteria = "BIC"):
        s = (self.BETA != 0).sum(axis = 0)
        if criteria == "BIC":
            output = 2*self.LOSS + np.log(self.sample_size)*s
        
        return np.argmin(output)
    
    def predict(self, data_x):
        best_model_id = self._select_best_model()
        #prepare data for linear part 
        linear_data_x = torch.tensor(data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_linear_data_list = standardize(linear_data_x)
        std_linear_data_x = std_linear_data_list['X']
        
        #prepare data for nonlinear part
        nonlinear_data_x = torch.tensor(data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_nonlinear_data_list = standardize(nonlinear_data_x)
        std_nonlinear_data_x = std_nonlinear_data_list['X']
        
        best_beta = torch.tensor(self.BETA[:,[best_model_id]],dtype = torch.double)
        linear_predict = torch.matmul(std_linear_data_x, best_beta)
        
        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_nonlinear_data_x).data
        best_dnn.train()
        risk_scores = linear_predict + nonlinear_predict
        
        return risk_scores.data.numpy().squeeze()
        
    
    def score(self,data_x,y):
        y_pred =  self.predict(data_x)
        y_event = y.iloc[:,1].values.astype(bool)
        y_time = y.iloc[:,0].values.astype('float64')
        c = concordance_index_censored(y_event,y_time,y_pred)[0]
        return c
    
    def estimate_nonlinear(self, data_x):
        best_model_id = self._select_best_model()
        
        #prepare data for nonlinear part
        nonlinear_data_x = torch.tensor(data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_nonlinear_data_list = standardize(nonlinear_data_x)
        std_nonlinear_data_x = std_nonlinear_data_list['X']
        
        
        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_nonlinear_data_x).data
        best_dnn.train()
        
        return nonlinear_predict
    
    def estimate_linear(self, data_x):
        best_model_id = self._select_best_model()
        #prepare data for linear part 
        linear_data_x = torch.tensor(data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_linear_data_list = standardize(linear_data_x)
        std_linear_data_x = std_linear_data_list['X']
        
        
        best_beta = torch.tensor(self.BETA[:,[best_model_id]],dtype = torch.double)
        linear_predict = torch.matmul(std_linear_data_x, best_beta)
        
        
        return linear_predict
    
class DPLC_Non_Linear:
    def __init__(self, linear_dim, non_linear_dim,
                 neuron1, neuron2,
                 dropout1,dropout2,
                 scad_LAMBDA,scad_a = 3.7,
                 learning_rate = 0.01, weight_decay = 0.1,
                 dnn_epoch = 100, scad_dnn_epoch = 10):
        
        self.linear_dim = linear_dim
        self.non_linear_dim = non_linear_dim
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        self.scad_a = scad_a
        self.scad_LAMBDA = scad_LAMBDA
        self.weight_decay = weight_decay
        self.dnn_epoch = dnn_epoch
        self.scad_dnn_epoch = scad_dnn_epoch
        
    def get_params(self,deep=True):
        return {"linear_dim" : self.linear_dim,
                "non_linear_dim" : self.non_linear_dim,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scad_a": self.scad_a,
                "scad_lam": self.scad_LAMBDA,
                "dnn_epoch": self.dnn_epoch,
                "scad_dnn_epoch": self.scad_dnn_epoch,
                "neuron1": self.neuron1,
                "neuron2": self.neuron2,
                "dropout1": self.dropout1,
                "dropout2": self.dropout2}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,data_x,data_y):
        self.sample_size = data_x.shape[0]
        t_ord = np.argsort(data_y.iloc[:,0])
        order_data_x = data_x.iloc[t_ord,:]
        order_data_y = torch.tensor(data_y.iloc[t_ord,:].to_numpy(),
                                           dtype = torch.double)
        
        #prepare data for linear part 
        order_linear_data_x = torch.tensor(order_data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_order_linear_data_list = standardize(order_linear_data_x)
        std_order_linear_data_x = std_order_linear_data_list['X']
        self.center = std_order_linear_data_list['center']
        self.scale = std_order_linear_data_list['scale']
        
        #prepare data for nonlinear part
        order_nonlinear_data_x = torch.tensor(order_data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_order_nonlinear_data_list = standardize(order_nonlinear_data_x)
        std_order_nonlinear_data_x = std_order_nonlinear_data_list['X']
        
        X_initial_beta = np.concatenate((std_order_linear_data_x.numpy(),
                                         std_order_nonlinear_data_x.numpy()),
                                        axis = 1)

        y_initial_beta = np.zeros_like(order_data_y.numpy())
        y_initial_beta[:,0] = order_data_y.numpy()[:,1]
        y_initial_beta[:,1] = order_data_y.numpy()[:,0]

        censor_initial_beta = np.core.records.fromarrays(
            y_initial_beta.transpose(), names='Status, Survival_in_days',
            formats = 'bool, f8')
        alpha = list(np.exp(np.linspace(np.log(1),np.log(0.01),10)))
        est_initial_beta = CoxnetSurvivalAnalysis(l1_ratio=1,alphas = alpha).fit(X_initial_beta,
                                                       censor_initial_beta)
        beta0_np = np.expand_dims(est_initial_beta._get_coef(None)[0][:self.linear_dim],axis = 1)
        beta0 = torch.from_numpy(beta0_np)
        beta0 = beta0.type(torch.double)
        
        #set up model
        lambda_len = len(self.scad_LAMBDA)
        self.LOSS = np.zeros((lambda_len,),dtype = np.float64)
        self.BETA = np.zeros((self.linear_dim,lambda_len),dtype = np.float64)
        self.MODEL = []
        lam_id = 0

        for l in self.scad_LAMBDA:
            loss = []
            beta = beta0#torch.zeros(self.linear_dim,1,dtype = torch.double)
            dnn  = Net_nonlinear(self.non_linear_dim,self.neuron1,self.neuron2,
                                 self.dropout1, self.dropout2,1) 
            
            best_dnn = Net_nonlinear(self.non_linear_dim,self.neuron1,self.neuron2,
                                 self.dropout1, self.dropout2,1) 
            best_beta = torch.zeros_like(beta)
            for i in range(self.scad_dnn_epoch):
                ## fit nonlinear DNN model
                linear_predict = torch.matmul(std_order_linear_data_x, beta)
                deep_pl_dnn_fit(std_order_nonlinear_data_x,
                                linear_predict,
                                order_data_y, 
                                dnn, 
                                self.weight_decay,
                                self.learning_rate,
                                self.dnn_epoch)
                
                ## fit linear model
                dnn.eval()
                nonlinear_predict = dnn.forward(std_order_nonlinear_data_x).data
                dnn.train()
                beta = deep_pl_scad_fit(std_order_linear_data_x, 
                                        order_data_y[:,1], 
                                        beta, nonlinear_predict, 
                                        l, self.linear_dim)
                
                linear_predict = torch.matmul(std_order_linear_data_x, beta)
                loss.append(coxph_loss_scad_like(nonlinear_predict + linear_predict,
                                 order_data_y))
                if loss[-1] == min(loss):
                    best_dnn.load_state_dict(dnn.state_dict())
                    best_beta = beta
                else:
                    # dnn.load_state_dict(best_dnn.state_dict())
                    beta = best_beta
                # print(beta[:10,:])   

            self.LOSS[lam_id] = min(loss).item()
            self.BETA[:,lam_id] = best_beta.numpy().squeeze()
            self.MODEL.append(best_dnn)
            # print(best_beta.numpy())
            lam_id = lam_id + 1
            print(lam_id)
            
        return self
    
    def _select_best_model(self,criteria = "AIC"):
        s = (self.BETA != 0).sum(axis = 0)
        if criteria == "BIC":
            output = 2*self.LOSS + np.log(self.sample_size)*s
        elif criteria == "AIC":
            output = 2*self.LOSS + 2*s
        elif criteria == "EBIC":
            output = 2*self.LOSS + (np.log(self.sample_size) +  np.log(self.linear_dim))*s
        return np.argmin(output)
    
    def predict(self, data_x,criteria):
        best_model_id = self._select_best_model(criteria)
        #prepare data for linear part 
        linear_data_x = torch.tensor(data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_linear_data_list = standardize(linear_data_x)
        std_linear_data_x = std_linear_data_list['X']
        
        #prepare data for nonlinear part
        nonlinear_data_x = torch.tensor(data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_nonlinear_data_list = standardize(nonlinear_data_x)
        std_nonlinear_data_x = std_nonlinear_data_list['X']
        
        best_beta = torch.tensor(self.BETA[:,[best_model_id]],dtype = torch.double)
        linear_predict = torch.matmul(std_linear_data_x, best_beta)
        
        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_nonlinear_data_x).data
        best_dnn.train()
        risk_scores = linear_predict + nonlinear_predict
        
        return risk_scores.data.numpy().squeeze()
        
    
    def score(self,data_x,y,criteria = "AIC"):
        y_pred =  self.predict(data_x,criteria)
        y_event = y.iloc[:,1].values.astype(bool)
        y_time = y.iloc[:,0].values.astype('float64')
        c = concordance_index_censored(y_event,y_time,y_pred)[0]
        return c
    
    
    def estimate_nonlinear(self, data_x,criteria):
        best_model_id = self._select_best_model(criteria)
        
        #prepare data for nonlinear part
        nonlinear_data_x = torch.tensor(data_x.iloc[:,self.linear_dim:].to_numpy(),
                                           dtype = torch.double)
        
        std_nonlinear_data_list = standardize(nonlinear_data_x)
        std_nonlinear_data_x = std_nonlinear_data_list['X']
        
        
        best_dnn = self.MODEL[best_model_id]
        best_dnn.eval()
        nonlinear_predict = best_dnn.forward(std_nonlinear_data_x).data
        best_dnn.train()
        
        return nonlinear_predict.data.numpy()
    
    def estimate_linear(self, data_x,criteria):
        best_model_id = self._select_best_model(criteria)
        #prepare data for linear part 
        linear_data_x = torch.tensor(data_x.iloc[:,:self.linear_dim].to_numpy(),
                                           dtype = torch.double)
        
        std_linear_data_list = standardize(linear_data_x)
        std_linear_data_x = std_linear_data_list['X']
        
        
        best_beta = torch.tensor(self.BETA[:,[best_model_id]],dtype = torch.double)
        linear_predict = torch.matmul(std_linear_data_x, best_beta)
        
        return linear_predict.data.numpy()



if __name__ == '__init__':
    from utils_plCox import survival_nonlinear_simulator,survival_linear_simulator
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    import pandas as pd
    import os
    
    
    p_no_select = 8
    p_select = 500
    N = 500
    s = 10
    censor_rate = 0.3
    beta_l = -0.5
    beta_u = 0.5
    alpha_l = -0.5
    alpha_u = 0.5
    normalSd = 1
    scaleData = True
    lam = 1/365
    
    data_path = 'D:\\study\\2021autumn\\LungCancer\\DFS\\experiment_debug\\data'
    
    X_scale, Z_scale, survCensor_numeric,beta,alpha, cstat = \
        survival_linear_simulator(p_select,p_no_select,N,s,censor_rate,
                                     lam,beta_l,beta_u,
                                     alpha_l,alpha_u,
                                     normalSd, scaleData)
    
    np.savetxt(os.path.join(data_path,'X.csv'), X_scale,delimiter=',')
    np.savetxt(os.path.join(data_path,'Z.csv'), Z_scale,delimiter=',')
    np.savetxt(os.path.join(data_path,'Y.csv'), survCensor_numeric,delimiter=',')
    survCensor = np.core.records.fromarrays(survCensor_numeric.transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')
    
    survCenor_numeric_order = np.zeros_like(survCensor_numeric)
    survCenor_numeric_order[:,0] = survCensor_numeric[:,1]
    survCenor_numeric_order[:,1] = survCensor_numeric[:,0]
    
    
    data_x_z = np.concatenate((X_scale,Z_scale),axis = 1)
    data_y = pd.DataFrame(survCenor_numeric_order)
    data_x = pd.DataFrame(data_x_z)
    
    dplc = DPLC_Linear(10,8,[0.04,0.05],weight_decay=0.01)
    dplc.fit(data_x,data_y)  
    dplc.score(data_x,data_y)
    
    
    
    
    p_no_select = 8
    p_select = 500
    N = 1000
    s = 10
    censor_rate = 0.3
    beta_l = -0.5
    beta_u = 0.5
    normalSd = 1
    scaleData = True
    lam = 1/365
    
    data_path = 'D:\\study\\2021autumn\\LungCancer\\DFS\\experiment_debug\\data'

    X_scale, Z_scale, survCensor_numeric,beta, cstat = \
        survival_nonlinear_simulator(p_select,p_no_select,N,s,censor_rate,
                                     lam,beta_l,beta_u,
                                     0.4,
                                     normalSd, scaleData)
        

    
    np.savetxt(os.path.join(data_path,'x.csv'), X_scale[1:500,:],delimiter=',')
    np.savetxt(os.path.join(data_path,'z.csv'), Z_scale[1:500,:],delimiter=',')
    np.savetxt(os.path.join(data_path,'y.csv'), survCensor_numeric[1:500,:],delimiter=',')
    
    np.savetxt(os.path.join(data_path,'x_test.csv'), X_scale[500:,:],delimiter=',')
    np.savetxt(os.path.join(data_path,'z_test.csv'), Z_scale[500:,:],delimiter=',')
    np.savetxt(os.path.join(data_path,'y_test.csv'), survCensor_numeric[500:,:],delimiter=',')
    
    
    survCensor = np.core.records.fromarrays(survCensor_numeric.transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')
    
    survCenor_numeric_order = np.zeros_like(survCensor_numeric)
    survCenor_numeric_order[:,0] = survCensor_numeric[:,1]
    survCenor_numeric_order[:,1] = survCensor_numeric[:,0]
    
    
    data_x_z = np.concatenate((X_scale,Z_scale),axis = 1)
    data_y = pd.DataFrame(survCenor_numeric_order)
    data_x = pd.DataFrame(data_x_z)
    
    dplc = DPLC_Non_Linear(500,8,8,4, 0.3, 0.3,[0.05],weight_decay=0.01,dnn_epoch = 50)
    dplc.fit(data_x,data_y)  
    dplc.score(data_x,data_y)
    dplc.BETA[:10,:]
    dplc._select_best_model()
    
    (dplc.BETA[10:,0] == 0).sum()

    dplc.score(data_x,data_y,'BIC')
