# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:26:28 2022

@author: sunym
"""
import numpy as np

import torch


def _make_riskset(time):
    assert time.dim() == 1, 'expected 1D array'
    
    n = time.size(0)
    _,indices = torch.sort(time,descending=True,stable = True)
    
    risk_set = torch.zeros(n,n,dtype = torch.bool)
    for i_org, i_sort in enumerate(indices):
        ti = time[i_sort].item()
        k = i_org
        while k < n and ti == time[indices[k]].item():
            k += 1
        risk_set[i_sort, indices[:k]] = True
    return risk_set

def logsumexp_masked(risk_scores, mask,
                     axis=1, keepdims=None):
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    mask_f = mask.type(risk_scores.type())
    
    risk_scores = torch.transpose(risk_scores, 0, 1)
    risk_scores_masked = torch.mul(risk_scores, mask_f)
    
    # for numerical stability, substract the maximum value
    # before taking the exponential
    amax,_ = torch.max(risk_scores_masked,axis = axis, keepdim=True)

    risk_scores_shift = risk_scores_masked - amax

    exp_masked = torch.mul(torch.exp(risk_scores_shift), mask_f)
    exp_sum = torch.sum(exp_masked, axis=axis, keepdims=True)
    output = amax + torch.log(exp_sum)
    return output

def coxph_loss_(event, riskset, predictions):
    # move batch dimension to the end so predictions get broadcast
    # row-wise when multiplying by riskset
    # compute log of sum over risk set for each row
    rr = logsumexp_masked(predictions, riskset, axis=1)
    event = event.type(rr.type())
    losses = torch.mul(event, rr - predictions)
    loss = torch.mean(losses)
    return loss

def coxph_loss(predictions,outcome):
    time = outcome[:,1]
    event = torch.unsqueeze(outcome[:,0],1)
    risk_set = _make_riskset(time)
    
    return coxph_loss_(event,risk_set,predictions)

def scad_penalty(beta,a = 3.7, lam = 1):
    beta_abs = torch.abs(beta)
    mask1 = (beta_abs <= lam).type(beta.type())
    mask2 = torch.logical_and(beta_abs > lam,beta_abs <= a * lam).type(beta.type())
    mask3 = (beta_abs > a * lam).type(beta.type())
    
    out1 = torch.mul(beta_abs,lam)
    out2 = -(torch.square(beta) - 2 * a * lam * beta_abs + lam * lam)/(2 * (a - 1))
    out3 = torch.ones_like(beta) * (a + 1) * lam * lam * 0.5
    
    out = torch.mul(mask1,out1) + torch.mul(mask2,out2) + torch.mul(mask3,out3)
    
    return torch.sum(out)

def coxph_loss_scad_like(predictions,outcome):
    predictions = torch.squeeze(predictions)
    n = predictions.size(0)
    haz = torch.zeros_like(predictions)
    rsk = torch.zeros_like(predictions)
    delta = outcome[:,0]
    Loss = torch.tensor(0,dtype = torch.double)
    for i in range(n): haz[i] = torch.exp(predictions[i])
    rsk[n-1] = haz[n-1]
    for i in range(n-2,-1,-1):
        rsk[i] = rsk[i+1] + haz[i]
    for i in range(n):
        Loss += delta[i]*predictions[i] - delta[i]*torch.log(rsk[i]) 
    
    return -1*Loss


if __name__ == '__main__':
    a = torch.Tensor([[1,-2,-3]])
    
    scad_penalty(a,a = 2)
    
    a_abs = torch.abs(a)
    torch.logical_and(a_abs > 1,a_abs <3)
    
