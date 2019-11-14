import numpy as np
import pickle
import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.optim as optim
from pyro.infer import SVI,JitTrace_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from preprocessor import to_pickle


def mcmc(data, ratings, model, mode="save",model_type):
    nuts_kernel = NUTS(model)
    hmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
    return hmc

def svi(data, ratings, model, guide, epoch):
    svi_model = SVI(model, 
                    guide, 
                    optim.Adam({"lr": .005}), 
                    loss=JitTrace_ELBO(), 
                    num_samples=500)

    pyro.clear_param_store()
    loss_list = []
    for i in range(epoch):
        ELBO = svi_model.step(data, ratings)
        if i % 500 == 0:
            print(ELBO)
            loss_list.append(ELBO)
    
    return svi_model,loss_list

