# adding feature test
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

class NeB:
    
    def model(self, data, ratings):

        with pyro.plate("betas", num_features + 1):
            betas = pyro.sample("beta", dist.Gamma(1,1))
            
        lambda_ = torch.exp(torch.sum(betas * data,axis=1))
        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Poisson(lambda_), obs = ratings)

        return y
    
    def guide(self, data, ratings):
        alphas_0 = pyro.param('weights_loc', torch.ones(data.shape[1]),  constraint=constraints.positive)
        alphas_1 = pyro.param('weights_scale', torch.ones(data.shape[1]), constraint=constraints.positive)               

        with pyro.plate("betas", data.shape[1]):
            betas = pyro.sample("beta", dist.Gamma(alphas_0, alphas_1))

