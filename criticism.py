import numpy as np
import pickle
import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.optim as optim
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO, TracePredictive
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from preprocessor import to_pickle


def svi_sampling(svi_model, data, ratings, model, model_type, mode="save"):
    
    def wrapped_model(data, ratings):
        pyro.sample("prediction", dist.Delta(model(data, ratings)))
   
    posterior = svi_model.run(data, ratings)
    sites = ["betas"]

    svi_samples = {site: EmpiricalMarginal(posterior, sites=site).
                   enumerate_support().detach().cpu().numpy() 
                   for site in sites}
    
   

    get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites) \
                   ._get_samples_and_weights()[0].detach().cpu().numpy()

    trace_pred = TracePredictive(wrapped_model,
                                 posterior,
                                 num_samples=500)
    post_pred = trace_pred.run(data, None)
    marginal = get_marginal(post_pred, ["prediction"])
    if mode == "save":
        to_pickle(marginal,"data_pickle/{}/{}_svi_samples".format(model_type, model_type))
        to_pickle(svi_samples,"data_pickle/{}/{}_svi_beta_dict".format(model_type, model_type))
    return marginal


        
def mcmc_posterior(hmc,data,ratings,model_type,mode="save"):
    hmc.run(data,ratings)
    hmc_beta_dict = {k: v.detach().cpu().numpy() for k, v in hmc.get_samples().items()}
    if mode == "save":
        to_pickle(hmc_beta_dict,"{}_mcmc_beta_dict".format(model_type))
    return hmc_beta_dict
        
def mlr_mcmc_sampling(betas, data, mode="save",is_cuda = False):
    if is_cuda:
      p_1 = torch.matmul(betas['beta_1'],data.T)
      p_1h = torch.matmul(betas['beta_1h'],data.T)
      p_2 = torch.matmul(betas['beta_2'], data.T)
      p_2h = torch.matmul(betas['beta_2h'], data.T)
      p_3 = torch.matmul(betas['beta_3'], data.T)
      p_3h = torch.matmul(betas['beta_3h'], data.T)
      p_4 = torch.matmul(betas['beta_4'], data.T)
      p_4h = torch.matmul(betas['beta_4h'], data.T)

    else:
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
    mci = softmax_array.mean(axis = 0)
    initial = True
    for i in range(300):
      if initial:
        initial = False
        y = torch.unsqueeze(pyro.sample("obs", dist.Categorical(probs=mci)),axis = 0)
      else:
        y = torch.cat([y, torch.unsqueeze(pyro.sample("obs", dist.Categorical(probs=mci)),axis = 0)],axis=0)
    if mode == "save":
      to_pickle(y,"mlr_mcmc_samples")
    return y
    