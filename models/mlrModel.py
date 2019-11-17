import numpy as np
import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.optim as optim
from pyro.infer import SVI,JitTrace_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS


class MlrModel:

    def model(self, data, ratings):
        #make one-hot vector length betas
        mu = torch.rand((8,data.shape[1]),dtype=torch.float)
        mu[4] /= 10
        mu[5] /= 10
        mu[6] /= 10
        mu[1] *= 2
        mu[2] *= 2
        sigma = 1
        with pyro.plate("betas", data.shape[1]):

            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma))
            #beta_5 = pyro.sample("beta_5", dist.Normal(mu, sigma))
        p_1 = torch.sum(beta_1 * data,axis=1)
        p_1h = torch.sum(beta_1h * data,axis=1)
        p_2 = torch.sum(beta_2 * data,axis=1)
        p_2h = torch.sum(beta_2h * data,axis=1)
        p_3 = torch.sum(beta_3 * data,axis=1)
        p_3h = torch.sum(beta_3h * data,axis=1)
        p_4 = torch.sum(beta_4 * data,axis=1)
        p_4h = torch.sum(beta_4h * data,axis=1)
        #p_5 = torch.sum(beta_5 * data,axis=1)

        p_array = torch.stack([p_1,p_1h,p_2,p_2h,p_3,p_3h,p_4,p_4h])
        exp_sum = torch.sum(torch.exp(p_array),axis=0)

        softmax_array=(torch.exp(p_array) / (1+exp_sum))
        temp_total = torch.sum(softmax_array, axis = 0)
        last_par = torch.unsqueeze(1 - temp_total, dim = 0)
        last_par[last_par < 0] = 0
        softmax_array = torch.cat([softmax_array, last_par], axis = 0).T

        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Categorical(probs=softmax_array), obs = ratings)
        return y


#     def mcmc(self, data, ratings, model, mode="save"):
#         nuts_kernel = NUTS(model)
#         hmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
#         hmc.run(data,ratings)
#         hmc_beta_dict = {k: v.detach().cpu().numpy() for k, v in hmc.get_samples().items()}
#         if mode == "save":
#             to_pickle(hmc,"mlr_hmc")
#             to_pickle(hmc_beta_dict,"mlr_hmc_beta_dict")
#         return hmc_beta_dict
    
        
#     def test(self, betas, data, mode="save"):
#         p_1 = torch.matmul(torch.from_numpy(betas['beta_1']),data.T)
#         p_1h = torch.matmul(torch.from_numpy(betas['beta_1h']),data.T)
#         p_2 = torch.matmul(torch.from_numpy(betas['beta_2']), data.T)
#         p_2h = torch.matmul(torch.from_numpy(betas['beta_2h']), data.T)
#         p_3 = torch.matmul(torch.from_numpy(betas['beta_3']), data.T)
#         p_3h = torch.matmul(torch.from_numpy(betas['beta_3h']), data.T)
#         p_4 = torch.matmul(torch.from_numpy(betas['beta_4']), data.T)
#         p_4h = torch.matmul(torch.from_numpy(betas['beta_4h']), data.T)

#         p_array = torch.exp(torch.stack([p_1,p_1h,p_2,p_2h,p_3,p_3h,p_4,p_4h], axis=1))
#         exp_sum = torch.sum(p_array,axis=1)
#         exp_sum_inv = torch.unsqueeze(1/(exp_sum + 1), axis = 1)
#         for i in range(3):
#             exp_sum_inv = torch.cat([exp_sum_inv,exp_sum_inv], axis = 1)

#         softmax_array=(p_array * exp_sum_inv)

#         temp_total = torch.sum(softmax_array, axis = 1)
#         last_par = torch.unsqueeze(1 - temp_total, dim = 1)
#         last_par[last_par < 0] = 0
#         softmax_array = torch.cat([softmax_array, last_par], axis = 1).transpose(1,2)

#         y = pyro.sample("obs", dist.Categorical(probs=softmax_array))
#         if mode == "save":
#             to_pickle(y,"mlr_test_samples")
#         return y
    
    
    def guide(self, data, ratings):
        sigma = pyro.param('sigma', torch.rand(data.shape[1]),  constraint=constraints.positive)
#         mu = pyro.param('mu', torch.zeros(data.shape[1]))     
        
        #mu = pyro.param('mu', torch.rand(data.shape[1]))        
        mu = pyro.param('mu', torch.rand((8,data.shape[1]),dtype=torch.float))
        print(mu)
        

        with pyro.plate("betas", data.shape[1]):
            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma))
            
        