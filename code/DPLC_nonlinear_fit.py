# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:37:57 2022

@author: sunym
"""
import torch 
from partialLik_plcox import coxph_loss,coxph_loss_scad_like

def deep_pl_dnn_fit(x,offset,y, model, weight_decay,learning_rate,epoch):
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'weight_decay': weight_decay},
       ], lr=learning_rate)
    
    for i in range(epoch):
        pred = model(x)
        pred = pred + offset
        loss = coxph_loss_scad_like(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
