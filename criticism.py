import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from preprocessor import to_pickle
from pyro.infer.predictive import Predictive


def svi_posterior(svi_model, data, ratings, model, model_type, sites,
                  if_save=True):

    def wrapped_model(data, ratings):
        pyro.sample("prediction", dist.Delta(model(data, ratings)))

    posterior = svi_model.run(data, ratings)
#     sites = ["beta_1","beta_1h","beta_2","beta_2h","beta_3",
#              "beta_3h","beta_4","beta_4h","beta_5"]

    svi_samples = {site: EmpiricalMarginal(posterior, sites=site).
                   enumerate_support().detach().cpu().numpy()
                   for site in sites}

    if if_save:
        to_pickle(svi_samples,
                  "data_pickle/{}_svi_beta_dict".format(model_type))
    return svi_samples


def mcmc_posterior(mcmc_model, data, ratings, model_type, if_save=True,
                   is_cuda=False):
    mcmc_model.run(data, ratings)
    if is_cuda:
        mcmc_beta_dict = {k: v.detach().cpu().numpy()
                          for k, v in mcmc_model.get_samples().items()}
    else:
        mcmc_beta_dict = mcmc_model.get_samples()
    if if_save:
        to_pickle(mcmc_beta_dict,
                  "data_pickle/{}_mcmc_beta_dict".format(model_type))
    return mcmc_beta_dict


def predictive_sampling(data, model, betas):

    res = Predictive(model, betas)
    post_sample = res.forward(data, None)

    return post_sample['obs']
