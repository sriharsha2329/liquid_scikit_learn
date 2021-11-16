# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:13:25 2021

@author: sriharsha
"""

import torch
from torch import nn
import numpy as np

#pre training
def logistic_regression(X,y,alpha):
    X=torch.from_numpy(X.astype('float32'))
    y=torch.from_numpy(y.astype('float32'))
    m=nn.Softmax()
    W = torch.randn(len(X[0]),requires_grad=True)
    pred = torch.matmul(W,X)
    pred=m(pred)
    loss = torch.multiply(y,torch.log(pred))
    loss=torch.sum(loss)
    loss.backward()
    with torch.no_grad():
        W=W+alpha*(W.grad)
    return W.numpy()


#post training
def liquid_logistic_regression(X,y,W,alpha,max_rate):
    X=torch.from_numpy(X.astype('float32'))
    y=torch.from_numpy(y.astype('float32'))
    W=torch.from_numpy(W)
    W_initial=W
    m=nn.Softmax()
    tau = torch.randn(len(X[0]), requires_grad=True)
    a=torch.rand(1, requires_grad=True)
    A=torch.randn(len(X[0]), requires_grad=True)
    M=m(torch.matmul(a*torch.ones(len(X[0])),X)+(1-a)*W)
    M=max_rate*M
    for i in np.arange(100):
        W=W+0.01*(-torch.multiply(tau+M,W)+torch.multiply(A,M))
    pred = torch.matmul(W,X)
    m=nn.Softmax()
    pred=m(pred)
    loss = torch.multiply(y,torch.log(pred))
    loss=torch.sum(loss)
    loss.backward()
    with torch.no_grad():
        tau=tau+alpha*(tau.grad)
        a=a+alpha*(a.grad)
        A=A+alpha*(A.grad)
    M=m(torch.matmul(a*torch.ones(len(X[0])),X)+(1-a)*W)
    M=max_rate*M
    W=W_initial
    for i in np.arange(100):
        W=W+0.01*(-torch.multiply(tau+M,W)+torch.multiply(A,M))
    return W.detach().numpy(),tau.numpy(),a.numpy(), A.numpy(),max_rate

def model_failure(W_initial,A,W,max_rate):
    for i,v in enumerate(W):
        w_min=np.min([W_initial[i],max_rate*A[i]])
        w_max=np.max([W_initial[i],max_rate*A[i]])
        if ((w_min<W[i]) and (w_max>W[i])):
            print("[{},{},{}], feature working properly".format(w_min,v,w_max))
        else:
            print("[{},{},{}], feature outdated, retrain the model".format(w_min,v,w_max))

