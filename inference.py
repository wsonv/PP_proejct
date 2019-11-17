import numpy as np
import pickle
import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.optim as optim
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO, TracePredictive, JitTraceEnum_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from preprocessor import to_pickle


def mcmc(data, ratings, model, model_type, mode="save"):
    nuts_kernel = NUTS(model)
    hmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
    if mode == "save":
        to_pickle(hmc,"{}_hmc".format(model_type))
    return hmc

def svi(data, ratings, model, guide, epoch):
#     svi_model = SVI(model, 
#                     guide, 
#                     optim.Adam({"lr": .005}), 
#                     loss=JitTrace_EnumELBO(), 
#                     num_samples=500)

#     pyro.clear_param_store()
#     loss_list = []
#     for i in range(epoch):
#         ELBO = svi_model.step(data, ratings)
#         if i % 500 == 0:
#             print(ELBO)
#             loss_list.append(ELBO)
    
#     return svi_model,loss_list
    elbo = JitTraceEnum_ELBO(max_plate_nesting=1)
    svi_model = SVI(model, 
                    guide, 
                    optim.Adam({"lr": .005}), 
                    loss=elbo, 
                    num_samples=500)

    pyro.clear_param_store()
    loss_list = []
    for i in range(epoch):
        ELBO = svi_model.step(data, ratings)
        
        if i % 100 == 0:
            loss_list.append(ELBO)
        if i % 500 == 0:
            print(ELBO)

    return svi_model, loss_list



