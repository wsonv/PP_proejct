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
#     def __init__(self):
#         self.vf = []
#         self.lat_len = np.arange(242)
#         f1 = lambda x, mu, sigma:  pyro.sample("beta_1_{}".format(x), dist.Normal(mu, sigma))
#         f2 = lambda x, mu, sigma:  pyro.sample("beta_1h_{}".format(x), dist.Normal(mu, sigma))
#         f3 = lambda x, mu, sigma:  pyro.sample("beta_2_{}".format(x), dist.Normal(mu, sigma))
#         f4 = lambda x, mu, sigma:  pyro.sample("beta_2h_{}".format(x), dist.Normal(mu, sigma))
#         f5 = lambda x, mu, sigma:  pyro.sample("beta_3_{}".format(x), dist.Normal(mu, sigma))
#         f6 = lambda x, mu, sigma:  pyro.sample("beta_3h_{}".format(x), dist.Normal(mu, sigma))
#         f7 = lambda x, mu, sigma:  pyro.sample("beta_4_{}".format(x), dist.Normal(mu, sigma))
#         f8 = lambda x, mu, sigma:  pyro.sample("beta_4h_{}".format(x), dist.Normal(mu, sigma))
#         fs = [f1,f2,f3,f4,f5,f6,f7,f8]
#         for f in fs:
#             self.vf.append(np.vectorize(f))
        
    # def model(self, data, ratings):
    #     #make one-hot vector length betas
        
    #     with pyro.plate("betas", data.shape[1]):
    #         mu = 0
    #         sigma = 3

    #         beta_1 = pyro.sample("beta_1", dist.Normal(mu, sigma))
    #         beta_1h = pyro.sample("beta_1h", dist.Normal(mu, sigma))
    #         beta_2 = pyro.sample("beta_2", dist.Normal(mu, sigma))
    #         beta_2h = pyro.sample("beta_2h", dist.Normal(mu, sigma))
    #         beta_3 = pyro.sample("beta_3", dist.Normal(mu, sigma))
    #         beta_3h = pyro.sample("beta_3h", dist.Normal(mu, sigma))
    #         beta_4 = pyro.sample("beta_4", dist.Normal(mu, sigma))
    #         beta_4h = pyro.sample("beta_4h", dist.Normal(mu, sigma))
    #         #beta_5 = pyro.sample("beta_5", dist.Normal(mu, sigma))
        
    #     p_1 = torch.sum(beta_1 * data,axis=1)
    #     p_1h = torch.sum(beta_1h * data,axis=1)
    #     p_2 = torch.sum(beta_2 * data,axis=1)
    #     p_2h = torch.sum(beta_2h * data,axis=1)
    #     p_3 = torch.sum(beta_3 * data,axis=1)
    #     p_3h = torch.sum(beta_3h * data,axis=1)
    #     p_4 = torch.sum(beta_4 * data,axis=1)
    #     p_4h = torch.sum(beta_4h * data,axis=1)
    #     #p_5 = torch.sum(beta_5 * data,axis=1)

    #     p_array = torch.stack([p_1,p_1h,p_2,p_2h,p_3,p_3h,p_4,p_4h])
    #     exp_sum = torch.sum(torch.exp(p_array),axis=0)

    #     softmax_array=(torch.exp(p_array) / (1+exp_sum))
    #     temp_total = torch.sum(softmax_array, axis = 0)
    #     last_par = torch.unsqueeze(1 - temp_total, dim = 0)
    #     last_par[last_par < 0] = 0
    #     softmax_array = torch.cat([softmax_array, last_par], axis = 0).T

    #     with pyro.plate("ratings", data.shape[0]):
    #         y = pyro.sample("obs", dist.Categorical(probs=softmax_array), obs = ratings)
    #     return y
    def model(self, data, ratings):
        #make one-hot vector length betas

        # mu = 10*torch.rand(8,data.shape[1]) - 5
        
        # to_deduct = torch.unsqueeze(torch.arange(8.) - 3, axis = 1)*0.5
        # to_deduct[-1] = to_deduct[2]
        # mu = mu + to_deduct
        # sigma = torch.ones(8,data.shape[1])*2


        # from preprocessor import load
        # beta_dict = load('/content/mlr_mcmc_beta_dict')
        # mu_1 = torch.from_numpy(np.average(beta_dict['beta_1'],axis = 0)).cuda()
        # mu_1h =  torch.from_numpy(np.average(beta_dict['beta_1h'],axis = 0)).cuda()
        # mu_2 = torch.from_numpy(np.average(beta_dict['beta_2'],axis = 0)).cuda()
        # mu_2h =  torch.from_numpy(np.average(beta_dict['beta_2h'],axis = 0)).cuda()
        # mu_3 = torch.from_numpy(np.average(beta_dict['beta_3'],axis = 0)).cuda()
        # mu_3h =  torch.from_numpy(np.average(beta_dict['beta_3h'],axis = 0)).cuda()
        # mu_4 = torch.from_numpy(np.average(beta_dict['beta_4'],axis = 0)).cuda()
        # mu_4h =  torch.from_numpy(np.average(beta_dict['beta_4h'],axis = 0)).cuda()
        mu = torch.zeros(8)
        sigma = torch.ones(8)
        with pyro.plate("betas", data.shape[1]):

            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))
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
#         for i in range(8):
#             betas[i] = torch.from_numpy(self.vf[i](self.lat_len,np.ones(self.lat_len.shape)*mu, np.ones(self.lat_len.shape)*sigma))

        #make one-hot vector length betas
#         beta_1 = torch.zeros(data.shape[1])
#         beta_1h = torch.zeros(data.shape[1])
#         beta_2 = torch.zeros(data.shape[1])
#         beta_2h = torch.zeros(data.shape[1])
#         beta_3 = torch.zeros(data.shape[1])
#         beta_3h = torch.zeros(data.shape[1])
#         beta_4 = torch.zeros(data.shape[1])
#         beta_4h = torch.zeros(data.shape[1])
#     def mcmc(self, data, ratings, model, mode="save"):
#         nuts_kernel = NUTS(model)
#         hmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
#         hmc.run(data,ratings)
#         hmc_beta_dict = {k: v.detach().cpu().numpy() for k, v in hmc.get_samples().items()}
#         if mode == "save":
#             to_pickle(hmc,"mlr_hmc")
#             to_pickle(hmc_beta_dict,"mlr_hmc_beta_dict")
#         return hmc_beta_dict
    
        
#     
    
    
#     def guide(self, data, ratings):
#         # sigma = pyro.param('sigma', torch.ones(data.shape[1]),  constraint=constraints.positive)
# #         mu = pyro.param('mu', torch.zeros(data.shape[1]))     
        
#         #mu = pyro.param('mu', torch.rand(data.shape[1]))        
#         # mu = pyro.param('mu', 4*torch.rand((data.shape[1]) - 2,dtype=torch.float))
        

#         with pyro.plate("betas", data.shape[1]):
#             mu = pyro.param('mu', 4*torch.rand(8) - 2)
#             sigma = pyro.param('sigma', torch.ones(8),  constraint=constraints.positive)
#             beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
#             beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
#             beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
#             beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
#             beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
#             beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
#             beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
#             beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))
            
    # def guide(self, data, ratings):
        # from preprocessor import load
        # beta_dict = load('/content/mlr_mcmc_beta_dict')
        # mu_1 = pyro.param("mu1", torch.from_numpy(np.average(beta_dict['beta_1'],axis = 0)).cuda())
        # mu_1h = pyro.param("mu1h", torch.from_numpy(np.average(beta_dict['beta_1h'],axis = 0)).cuda())
        # mu_2 = pyro.param("mu2", torch.from_numpy(np.average(beta_dict['beta_2'],axis = 0)).cuda())
        # mu_2h = pyro.param("mu2h", torch.from_numpy(np.average(beta_dict['beta_2h'],axis = 0)).cuda())
        # mu_3 = pyro.param("mu3", torch.from_numpy(np.average(beta_dict['beta_3'],axis = 0)).cuda())
        # mu_3h = pyro.param("mu3h", torch.from_numpy(np.average(beta_dict['beta_3h'],axis = 0)).cuda())
        # mu_4 = pyro.param("mu4", torch.from_numpy(np.average(beta_dict['beta_4'],axis = 0)).cuda())
        # mu_4h = pyro.param("mu4h", torch.from_numpy(np.average(beta_dict['beta_4h'],axis = 0)).cuda())
        # #mu = pyro.param("mu", 4*torch.rand(8,data.shape[1]) - 2)
        # sigma = pyro.param('sigma', torch.rand(8,data.shape[1]),  constraint=constraints.positive)
        # with pyro.plate("betas", data.shape[1]):
        #     # mu = pyro.param("mu", torch.tensor(0.))
        #     # sigma = pyro.param('sigma', torch.tensor(5.),  constraint=constraints.positive)
        #     beta_1 = pyro.sample("beta_1", dist.Normal(mu_1, sigma[0]))
        #     beta_1h = pyro.sample("beta_1h", dist.Normal(mu_1h, sigma[1]))
        #     beta_2 = pyro.sample("beta_2", dist.Normal(mu_2, sigma[2]))
        #     beta_2h = pyro.sample("beta_2h", dist.Normal(mu_2h, sigma[3]))
        #     beta_3 = pyro.sample("beta_3", dist.Normal(mu_3, sigma[4]))
        #     beta_3h = pyro.sample("beta_3h", dist.Normal(mu_3h, sigma[5]))
        #     beta_4 = pyro.sample("beta_4", dist.Normal(mu_4, sigma[6]))
        #     beta_4h = pyro.sample("beta_4h", dist.Normal(mu_4h, sigma[7]))


    def guide(self, data, ratings):

        # mu = pyro.param("mu", 10*torch.rand(8,data.shape[1]) - 5)
        
        # to_deduct = torch.unsqueeze(torch.arange(8.) - 3, axis = 1)*0.5
        # to_deduct[-1] = to_deduct[2]
        # mu = mu + to_deduct
        # sigma = pyro.param('sigma', torch.ones(8,data.shape[1]),  constraint=constraints.positive)*2
        mu = pyro.param("mu", torch.zeros(8, data.shape[1]))
        sigma = pyro.param('sigma', torch.ones(8,data.shape[1]),  constraint=constraints.positive)

        with pyro.plate("betas", data.shape[1]):
            # mu = pyro.param("mu", torch.tensor(0.))
            # sigma = pyro.param('sigma', torch.tensor(5.),  constraint=constraints.positive)
            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))






#         for i in range(8):
#             self.vf[i](self.lat_len, mu, sigma)
        
#         for i in range(data.shape[1]):
#             beta_1 = pyro.sample("beta_1_{}".format(i), dist.Normal(mu, sigma))
#             beta_1h = pyro.sample("beta_1h_{}".format(i), dist.Normal(mu, sigma))
#             beta_2 = pyro.sample("beta_2_{}".format(i), dist.Normal(mu, sigma))
#             beta_2h = pyro.sample("beta_2h_{}".format(i), dist.Normal(mu, sigma))
#             beta_3 = pyro.sample("beta_3_{}".format(i), dist.Normal(mu, sigma))
#             beta_3h = pyro.sample("beta_3h_{}".format(i), dist.Normal(mu, sigma))
#             beta_4 = pyro.sample("beta_4_{}".format(i), dist.Normal(mu, sigma))
#             beta_4h = pyro.sample("beta_4h_{}".format(i), dist.Normal(mu, sigma))