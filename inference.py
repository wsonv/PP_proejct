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


def svi_sampling(svi_model, data, ratings)
    posterior = svi_model.run(data, ratings)
    sites = ["betas"]

    svi_samples = {site: EmpiricalMarginal(posterior, sites=site).
                   enumerate_support().detach().cpu().numpy() 
                   for site in sites}

get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
def wrapped_model(data, ratings):
    pyro.sample("prediction", dist.Delta(model(data, ratings)))
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=500)
post_pred = trace_pred.run(data, None)
marginal = get_marginal(post_pred, ["prediction"])



        svi_samples = {site: EmpiricalMarginal(posterior, sites=site)
                             .enumerate_support().detach().cpu().numpy()
                       for site in sites}

        get_marginal = lambda traces, 
                       sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
       
        trace_pred = TracePredictive(wrapped_model,
                                     posterior,
                                     num_samples=500)
        post_pred = trace_pred.run(data, None)
        marginal = get_marginal(post_pred, ["prediction"])

def wrapped_model(data, ratings, model):
        pyro.sample("prediction", dist.Delta(model(data, ratings)))