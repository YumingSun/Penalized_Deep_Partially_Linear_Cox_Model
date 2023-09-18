# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:24:11 2022

@author: sunym
"""
import torch


def standardize(X):
    center = torch.mean(X,0,True)
    X_center = X - center
    
    scale = torch.pow(torch.mean(torch.pow(X_center,2),0,True),0.5)
    X_std = X_center/scale
    
    return {'X': X_std, 'center': center, 'scale': scale}

def wcrossprod(X,y, w, n, j):
    val = X[0,j] * y[0] * w[0]
    for i in range(1,n):
        val += X[i,j] * y[i] * w[i]
    return val
    
def wsqsum(X,w,n,j):
    val = torch.pow(X[0,j],2) * w[0]
    for i in range(1,n):
        val += torch.pow(X[i,j],2) * w[i]
    return val


def SCAD(z,lam,gamma, v):
    if z > 0:
        s = 1
    else:
        s = -1
    
    if torch.abs(z) <= lam:
        return 0
    elif torch.abs(z) <= 2*lam:
        return s * (torch.abs(z) - lam)/v
    elif torch.abs(z) <= gamma*lam:
        return s*(torch.abs(z) - gamma*lam/(gamma - 1))/(v*(1 - 1/(gamma -1)))
    else:
        return z/v
    
        
    
def deep_pl_scad_fit(X, delta, beta0, offset, lam, dfmax, eps=1e-4, max_iter=10000,gamma = 3.7):
    n = X.shape[0]
    p = X.shape[1]
    L = 1
    
    tot_iter = 0
    beta = torch.zeros(p,L,dtype = torch.double)
    Loss = torch.zeros(L,dtype = torch.double)
    iter = torch.zeros(L,dtype = torch.int32)
    Eta = torch.zeros(n,L, dtype = torch.double)
    
    a = torch.zeros(p,dtype = torch.double)
    haz  = torch.zeros(n,dtype = torch.double)
    rsk = torch.zeros(n,dtype = torch.double)
    r = torch.zeros(n,dtype = torch.double)
    h = torch.zeros(n,dtype = torch.double)
    e = torch.zeros(p,dtype = torch.int8)
    eta = torch.matmul(X, beta0) + offset
    eta = torch.squeeze(eta)

    
    for l in range(L):
        while tot_iter < max_iter:
            while tot_iter < max_iter:
                iter[l] += 1
                tot_iter += 1
                Loss[l] = 0
                maxChange = 0
                
                for i in range(n): haz[i] = torch.exp(eta[i])
                rsk[n-1] = haz[n-1]
                for i in range(n-2,-1,-1):
                    rsk[i] = rsk[i+1] + haz[i]
                for i in range(n):
                    Loss[l] += delta[i]*eta[i] - delta[i]*torch.log(rsk[i])
                
                
                h[0] = delta[0]/rsk[0]
                for i in range(1,n):
                    h[i] = h[i-1] + delta[i]/rsk[i]
                for i in range(n):
                    h[i] = h[i]*haz[i]
                    s = delta[i] - h[i]
                    if h[i]==0:
                        r[i]=0
                    else:
                        r[i] = s/h[i]
                        
                for j in range(p):
                    if e[j]:
                        xwr = wcrossprod(X, r, h, n, j)
                        xwx = wsqsum(X, h, n, j)
                        u = xwr/n + (xwx/n)*a[j]
                        v = xwx/n
            
                        beta[j,l] = SCAD(u, lam,gamma, v)
                        
                        shift = beta[j,l]  - a[j]
                        if shift != 0:
                            for i in range(n):
                                si = shift*X[i,j]
                                r[i] -= si
                                eta[i] += si
                            if  torch.abs(shift)*torch.sqrt(v) > maxChange:
                                maxChange = torch.abs(shift)*torch.sqrt(v)
                
                for j in range(p):
                    a[j] = beta[j,l]
                if maxChange < eps:
                    break
                # print("iter: {}, maxChange: {}".format(tot_iter,maxChange))
                
                
            violations = 0
            for j in range(p):
                if e[j] == 0:
                    xwr = wcrossprod(X, r, h, n, j)/n
                    l1 = lam
                    if torch.abs(xwr) > l1:
                        e[j] = 1
                        violations += 1
            if violations == 0:
                for i in range(n):
                    Eta[i,l] = eta[i]
                break
            # print(violations)
    return beta