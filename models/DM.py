import numpy as np
import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.optim as optim
from pyro.infer import SVI,JitTrace_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS


class DMModel:
    
    
    def model(self, data, ratings):
        
        
        N = 9
        mu = 0
        sigma = 1
        
        beta_0 = pyro.sample("beta_0", dist.Normal(mu, sigma))
        beta_1 = pyro.sample("beta_1", dist.Normal(mu, sigma))
        beta_2 = pyro.sample("beta_2", dist.Normal(mu, sigma))
        beta_3 = pyro.sample("beta_3", dist.Normal(mu, sigma))
        beta_4 = pyro.sample("beta_4", dist.Normal(mu, sigma))
        beta_5 = pyro.sample("beta_5", dist.Normal(mu, sigma))
        beta_6 = pyro.sample("beta_6", dist.Normal(mu, sigma))
        beta_7 = pyro.sample("beta_7", dist.Normal(mu, sigma))
        beta_8 = pyro.sample("beta_8", dist.Normal(mu, sigma))
        
        alpha = [alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7,alpha_8]
        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.classDirichletMultinomial(concentration=alpha), obs = ratings)
        return y
        
        
#         #make one-hot vector length betas
#         mu = torch.rand((8,data.shape[1]),dtype=torch.float)
#         sigma = 1
#         with pyro.plate("betas", data.shape[1]):

#             beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma))
#             beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma))
#             beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma))
#             beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma))
#             beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma))
#             beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma))
#             beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma))
#             beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma))
#             #beta_5 = pyro.sample("beta_5", dist.Normal(mu, sigma))
#         p_1 = torch.sum(beta_1 * data,axis=1)
#         p_1h = torch.sum(beta_1h * data,axis=1)
#         p_2 = torch.sum(beta_2 * data,axis=1)
#         p_2h = torch.sum(beta_2h * data,axis=1)
#         p_3 = torch.sum(beta_3 * data,axis=1)
#         p_3h = torch.sum(beta_3h * data,axis=1)
#         p_4 = torch.sum(beta_4 * data,axis=1)
#         p_4h = torch.sum(beta_4h * data,axis=1)
#         #p_5 = torch.sum(beta_5 * data,axis=1)

#         p_array = torch.stack([p_1,p_1h,p_2,p_2h,p_3,p_3h,p_4,p_4h])
#         exp_sum = torch.sum(torch.exp(p_array),axis=0)

#         softmax_array=(torch.exp(p_array) / (1+exp_sum))
#         temp_total = torch.sum(softmax_array, axis = 0)
#         last_par = torch.unsqueeze(1 - temp_total, dim = 0)
#         last_par[last_par < 0] = 0
#         softmax_array = torch.cat([softmax_array, last_par], axis = 0).T

#         with pyro.plate("ratings", data.shape[0]):
#             y = pyro.sample("obs", dist.Categorical(probs=softmax_array), obs = ratings)

        

    
    
    def guide(self, data, ratings):
        sigma = pyro.param('signma', torch.rand(data.shape[1]),  constraint=constraints.positive)
#         mu = pyro.param('mu', torch.zeros(data.shape[1]))     
        
        #mu = pyro.param('mu', torch.rand(data.shape[1]))        
        mu = torch.rand((8,data.shape[1]),dtype=torch.float)


        with pyro.plate("betas", data.shape[1]):
            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma))
        