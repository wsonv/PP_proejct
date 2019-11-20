import pyro
import pyro.optim as optim
from pyro.infer import SVI, JitTraceEnum_ELBO
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from preprocessor import to_pickle


def mcmc(data, ratings, model, model_type, if_save=False, num_sample=200,
         num_w_steps=200):
    nuts_kernel = NUTS(model, jit_compile=True)
    hmc = MCMC(nuts_kernel, num_samples=num_sample, warmup_steps=num_w_steps)
    if if_save:
        to_pickle(hmc, "data_pickle/{}_mcmc_model".format(model_type))
    return hmc


def svi(data, ratings, model, guide, epoch, model_type, if_save=True,
        if_print=True, num_sample=200):
    elbo = JitTraceEnum_ELBO(max_plate_nesting=1)
    svi_model = SVI(model,
                    guide,
                    optim.Adam({"lr": .005}),
                    loss=elbo,
                    num_samples=num_sample)

    pyro.clear_param_store()
    loss_list = []
    for i in range(epoch):
        ELBO = svi_model.step(data, ratings)
        loss_list.append(ELBO)
        if i % 500 == 0 and if_print:
            print(ELBO)
    if if_save:
        to_pickle(loss_list, "data_pickle/{}_svi_loss".format(model_type))
    return svi_model, loss_list
