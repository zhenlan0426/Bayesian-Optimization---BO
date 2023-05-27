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
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
#from botorch.acquisition.analytic import ExpectedImprovement,ProbabilityOfImprovement,UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll # TODO fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel,MaternKernel,ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from torch.autograd import Function
from torch.nn.functional import one_hot


device = 'cpu'
dtype = torch.double

class Transform(object):
    def __init__(self,cat_feat,integer,bounds):
        # cat_feat {feature name : {feature val: int val},...}
        # bounds  [[low_0,high_0],[]...], Assume cont_feat are before cat_feat in parameter

        self.cat_feat = cat_feat
        self.integer = integer
        self.bounds = bounds
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

class Mean_std(AnalyticAcquisitionFunction):
    def __init__(self,model,beta) -> None:
        super().__init__(model=model)
        self.beta = beta

    def forward(self, X):
        mean, std = self._mean_and_sigma(X, compute_sigma=True)
        return mean - self.beta * std
    
def choose_best(model,bounds,x,y,T,type_,beta=0.5):
    if type_ == "existing":
        y_best = np.argmax(y)
        x_best = x.iloc[y_best]
        return x_best, y.iloc[y_best][0]
    if type_ == "mean":
        mean_fun = Mean_std(model,beta)
        x_best = optimize_acqf(mean_fun,bounds,q=1,num_restarts=1,raw_samples=1)[0].detach()
        return T.backward(x_best[0])

class RoundSTE(Function):
    r"""Apply a rounding function and use a ST gradient estimator."""

    @staticmethod
    def forward(ctx,input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

roundSTE = RoundSTE.apply

class OneHotArgmaxSTE(Function):
    r"""Apply a discretization (argmax) to a one-hot encoded categorical, return a one-hot encoded categorical, and use a STE gradient estimator."""
    @staticmethod
    def forward(ctx,input_):
        return one_hot(input_.argmax(dim=1))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

oneHotArgmaxSTE = OneHotArgmaxSTE.apply

# def hook_factory(T,STE):
#     d0 = len(T.bounds)
#     def pre_fun(module, x):
#         x = x[0] # input is a tuple
#         if STE:
#             for i in T.integer:
#                 x[i] = roundSTE(x[i])
#         out = [x[:,:d0],]
#         count0 = d0
#         for d_i in T.cat_shape:
#             sm = torch.softmax(x[:,count0:count0+d_i],1)
#             if STE:
#                 sm = oneHotArgmaxSTE(sm)
#             out.append(sm)
#             count0 += d_i
#         return torch.cat(out,1)
#     return pre_fun

def hook_factory(T,STE):
    d0 = len(T.bounds)
    def pre_softmax(module, x):
        x = x[0] # input is a tuple
        out = [x[:,:d0],]
        count0 = d0
        for d_i in T.cat_shape:
            out.append(torch.softmax(x[:,count0:count0+d_i],1))
            count0 += d_i
        return torch.cat(out,1)
    return pre_softmax
    
def BO(fun,x,y,T,\
       acq_fun,
       acq_kwargs,
       BaseKernel,
       eps,# f_best + eps
       STE,
       beta,
       q,
       num_restarts,
       raw_samples,
       Bo_iter,
       verbose):
    # x: DataFrame of inputs, y: DF of scores
    x_name = list(x.columns)
    x_tor,y_tor = T.forward(x,y)
    #b = 10 # bound for logit
    d0 = len(T.bounds)
    d1 = x_tor.shape[1] - d0
    pre_softmax = hook_factory(T,STE)
    #bounds = torch.tensor([[0.0] * d0 + [-b] * d1, [1.0] * d0 + [b] * d1], device=device, dtype=dtype)
    bounds = torch.tensor([[0.0] * (d0+d1), [1.0] * (d0+d1)], device=device, dtype=dtype)
    model = initialize_model(x_tor,y_tor, BaseKernel)
    for j in range(1,1+Bo_iter):
        # set up acqucision fun
        if 'best_f' in inspect.signature(acq_fun).parameters:
            acq_kwargs['best_f'] = y.max().item() + eps
        h = model.register_forward_pre_hook(pre_softmax)
        acq = acq_fun(model,**acq_kwargs)
        # optimize over x_next
        x_next = optimize_acqf(acq,bounds,q=q,num_restarts=num_restarts,raw_samples=raw_samples)[0].detach() # 1,d
        x_next = T.backward(x_next[0])
        h.remove()
        
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
    h = model.register_forward_pre_hook(pre_softmax)
    x_best_mean = choose_best(model,bounds,x,y,T,"mean",beta)
    h.remove()
    y_best_mean = fun(*x_best_mean)
    
    if y_best_exist < y_best_mean:
        return x_best_mean,y_best_mean,x,y,model
    else:
        return x_best_exist,y_best_exist,x,y,model 