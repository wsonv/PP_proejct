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

class PossModel:
    
    def model(self, data, ratings):

        with pyro.plate("betas", num_features + 1):
            betas = pyro.sample("beta", dist.Gamma(1,1))
            
        lambda_ = torch.exp(torch.sum(betas * data,axis=1))
        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Poisson(lambda_), obs = ratings)

        return y

    def mcmc(self, data, ratings, model, mode="save"):
        nuts_kernel = NUTS(model)
        hmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
        hmc.run(data, ratings)
        hmc_beta_dict = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
        
        if mode == "save":
            to_pickle(hmc,"poss_hmc")
            to_pickle(hmc_beta_dict,"poss_hmc_beta_dict")
        return hmc_beta_dict
        
    def test(self, betas, data, mode="save"):
        p_1 = torch.matmul(torch.from_numpy(betas['beta_1']),data.T)
        p_1h = torch.matmul(torch.from_numpy(betas['beta_1h']),data.T)
        p_2 = torch.matmul(torch.from_numpy(betas['beta_2']), data.T)
        p_2h = torch.matmul(torch.from_numpy(betas['beta_2h']), data.T)
        p_3 = torch.matmul(torch.from_numpy(betas['beta_3']), data.T)
        p_3h = torch.matmul(torch.from_numpy(betas['beta_3h']), data.T)
        p_4 = torch.matmul(torch.from_numpy(betas['beta_4']), data.T)
        p_4h = torch.matmul(torch.from_numpy(betas['beta_4h']), data.T)

        p_array = torch.exp(torch.stack([p_1,p_1h,p_2,p_2h,p_3,p_3h,p_4,p_4h], axis=1))
        exp_sum = torch.sum(p_array,axis=1)
        exp_sum_inv = torch.unsqueeze(1/(exp_sum + 1), axis = 1)
        for i in range(3):
            exp_sum_inv = torch.cat([exp_sum_inv,exp_sum_inv], axis = 1)

        softmax_array=(p_array * exp_sum_inv)

        temp_total = torch.sum(softmax_array, axis = 1)
        last_par = torch.unsqueeze(1 - temp_total, dim = 1)
        last_par[last_par < 0] = 0
        softmax_array = torch.cat([softmax_array, last_par], axis = 1).transpose(1,2)

        y = pyro.sample("obs", dist.Categorical(probs=softmax_array))
        if mode == "save":
            to_pickle(y,"poss_test_samples")
        return y
    
    
    def guide(self, data, ratings):
        alphas_0 = pyro.param('weights_loc', torch.ones(data.shape[1]),  constraint=constraints.positive)
        alphas_1 = pyro.param('weights_scale', torch.ones(data.shape[1]), constraint=constraints.positive)               

        with pyro.plate("betas", data.shape[1]):
            betas = pyro.sample("beta", dist.Gamma(alphas_0, alphas_1))
    
    def svi(self, data, ratings, model, guide, epoch):
        svi_model = SVI(model, 
          guide, 
          optim.Adam({"lr": .005}), 
          loss=JitTrace_ELBO(), 
          num_samples=500)

        pyro.clear_param_store()
        for i in range(epoch):
            ELBO = svi_model.step(data, ratings)
            if i % 500 == 0:
                print(ELBO)
        return svi_model



