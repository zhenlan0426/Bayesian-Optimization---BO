#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:45:38 2023

@author: will
"""
import numpy as np
import pandas as pd
import inspect
import torch
from botorch.models import SingleTaskGP
#from botorch.acquisition.analytic import ExpectedImprovement,ProbabilityOfImprovement,UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll # TODO fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel,MaternKernel,ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior

device = 'cpu'
dtype = torch.double

class Transform(object):
    def __init__(self,cat_feat,integer,bounds,IsMax):
        # cat_feat {feature name : {feature val: int val},...}
        # bounds  [[low_0,high_0],[]...], Assume cont_feat are before cat_feat in parameter
        # IsMax is True, we aim to max the fun
        self.cat_feat = cat_feat
        self.integer = integer
        self.bounds = bounds
        self.IsMax = IsMax
        self.cat_feat_inv = [{v: k for k,v in cat_feat[d].items()} for d in cat_feat]
        self.cat_shape = [len(cat_feat[k]) for k in cat_feat]
        
    def forward(self,parameter,score):
        # parameter: DataFrame -> x,y in [0,1]^d,R
        # cat features
        # str -> int
        out = []
        for cat in self.cat_feat:
            out.append((len(self.cat_feat[cat]),parameter[cat].map(self.cat_feat[cat]).values))
        # int -> one hot
        cat_out = []
        n = out[0][1].shape[0]
        for i,o in out:
            temp = np.zeros((n,i))
            temp[np.arange(n), o] = 1.0
            cat_out.append(temp)
        cat_out = np.concatenate(cat_out,1)
        # cont feature
        # cat features should be at the end of col
        cont_out = []
        temp = parameter.iloc[:,:len(self.bounds)].values
        for f,(l,h) in zip(temp.T,self.bounds):
            cont_out.append((f - l)/(h - l))
        cont_out = np.stack(cont_out,1)
        x = np.concatenate((cont_out,cat_out),1)
        y = (score.values - score.values.mean())/score.values.std()
        y = y if self.IsMax else -y
        return torch.Tensor(x,device=device).to(dtype),torch.Tensor(y,device=device).to(dtype)
    
    def backward(self,x):
        # x in [0,1]^d -> mixed int,float,str
        x = x.cpu().detach().numpy()
        ## cont_feature ##
        # unnormalize
        cont_out = []
        for f,(l,h) in zip(x,self.bounds):
            cont_out.append(l + (h-l)*f)
        # round for integer
        for i in self.integer:
            cont_out[i] = cont_out[i].astype(int)
        ## cat_feature ##
        n0 = len(cont_out)
        cat2int = []
        for d in self.cat_feat_inv:
            cat2int.append(np.argmax(x[n0:n0+len(d)]))
            n0 += len(d)
        cat2str = []
        for i,m in zip(cat2int,self.cat_feat_inv):
            cat2str.append(m[i])
        return cont_out + cat2str
    
    
    
    
# =============================================================================
# BO    
# =============================================================================
def initialize_model(x, y, BaseKernel, state_dict=None):
    covar_module = ScaleKernel(BaseKernel(ard_num_dims=x.shape[-1],
                                          lengthscale_prior=GammaPrior(3.0, 6.0)),
                               outputscale_prior=GammaPrior(2.0, 0.15))
    model = SingleTaskGP(x, y, covar_module=covar_module).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    fit_gpytorch_mll(mll);
    return model

def choose_best(model,bounds,x,y,T,type_):
    if type_ == "existing":
        y_best = np.argmax(y) if T.IsMax else np.argmin(y)
        x_best = x.iloc[y_best]
        return x_best, y.iloc[y_best][0]
    if type_ == "mean":
        # TODO: fix for softmax
        mean_fun = lambda x:model(x).mean
        x_best = optimize_acqf(mean_fun,bounds,q=1,num_restarts=1,raw_samples=1)[0].detach()
        return x_best

def BO(fun,x,y,T,\
       acq_fun,
       acq_kwargs,
       BaseKernel,
       q,
       num_restarts,
       raw_samples,
       Bo_iter,
       verbose):
    # x: DataFrame of inputs, y: DF of scores
    x_name = list(x.columns)
    x_tor,y_tor = T.forward(x,y)
    b = 10 # bound for logit
    d0 = len(T.bounds)
    d1 = x_tor.shape[1] - d0
    bounds = torch.tensor([[0.0] * d0 + [-b] * d1, [1.0] * d0 + [b] * d1], device=device, dtype=dtype)    
    model = initialize_model(x_tor,y_tor, BaseKernel)
    for j in range(1,1+Bo_iter):
        # set up acqucision fun
        if 'best_f' in inspect.signature(acq_fun).parameters:
            acq_kwargs['best_f'] = y.max().item()
        acq = acq_fun(model,**acq_kwargs)
        def acq2(x):
            out = [x[:,:d0],]
            count0 = d0
            for d_i in T.cat_shape:
                out.append(torch.softmax(x[:,count0:count0+d_i],1))
                count0 += d_i
            return acq(torch.cat(out,1))
        # optimize over x_next
        x_next = optimize_acqf(acq2,bounds,q=q,num_restarts=num_restarts,raw_samples=raw_samples)[0].detach() # 1,d
        x_next = T.backward(x_next[0])
        
        # try x_next
        y_next = fun(*x_next)
        x_next = pd.DataFrame([x_next],columns=x_name) # -> DF for pd.concat
        # update dataset
        x = pd.concat([x,x_next])
        y = pd.concat([y,pd.DataFrame([y_next,],columns=['scores'])])

        # update model
        x_tor,y_tor = T.forward(x,y)
        model = initialize_model(x_tor,y_tor, BaseKernel, model.state_dict())

        if j%verbose == 0:
            x_best,y_best = choose_best(model,bounds,x,y,T,"existing")
            print('best val is {} at iter {}'.format(y_best.item(),j))
    
    x_best_exist,y_best_exist = choose_best(model,bounds,x,y,T,"existing")
    x_best_mean = choose_best(model,bounds,x,y,T,"mean")
    y_best_mean =  fun(*T.backward(x_best_mean[0]))
    
    if (y_best_exist > y_best_mean) ^ T.IsMax:
        return x_best_mean,y_best_mean,x,y,model
    else:
        return x_best_exist,y_best_exist,x,y,model 