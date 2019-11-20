import numpy as np
import pyro
import torch
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, JitTrace_ELBO
from pyro.infer.predictive import Predictive as pred
from preprocessor import to_pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class FactorModel:

    def __init__(self, data_dim):
        self.data_num = data_dim[0]
        self.fea_num = data_dim[1]

    def lf_model(self, data):

        k = 20
        pre_loc = torch.zeros(self.data_num, k)
        pre_sca = torch.ones(self.data_num, k)
        with pyro.plate('zs', self.data_num, k):
            z = pyro.sample("z", dist.Normal(pre_loc, pre_sca))

        pre_theta_loc = torch.rand(self.fea_num, k) * 2 - 1
        pre_theta_sca = torch.ones(self.fea_num, k)
        with pyro.plate('thetas', self.fea_num, k):
            theta = pyro.sample("theta", dist.Normal(pre_theta_loc,
                                                     pre_theta_sca))
        A = torch.matmul(z, theta.T)
        A_exp = 1 / (1 + torch.exp(-A))
        with pyro.plate('As', A.shape[0], A.shape[1]):
            y = pyro.sample("a", dist.Bernoulli(A_exp), obs=data)
        return y

    def lf_guide(self, data):
        k = 20

        pre0 = pyro.param("alpha_0", torch.zeros(self.data_num, k))
        pre1 = pyro.param("alpha_1", torch.ones(self.data_num, k))
        with pyro.plate('zs', self.data_num, k):
            pyro.sample("z", dist.Normal(pre0, pre1))

        pre_theta_0 = pyro.param("talpha_0", torch.zeros(self.fea_num, k))
        pre_theta_1 = pyro.param("talpha_1", torch.ones(self.fea_num, k))
        with pyro.plate('thetas', self.fea_num, k):
            pyro.sample("theta", dist.Normal(pre_theta_0, pre_theta_1))

    def lf_inference(self, lf_data, lf_model, lf_guide, epoch):
        elbo = JitTrace_ELBO(max_plate_nesting=1)
        lf_svi_model = SVI(lf_model,
                           lf_guide,
                           optim.Adam({"lr": .005}),
                           loss=elbo,
                           num_samples=300)
        pyro.clear_param_store()
        loss_list = []
        for i in range(epoch):
            ELBO = lf_svi_model.step(lf_data)

            if i % 100 == 0:
                loss_list.append(ELBO)
            if i % 500 == 0:
                print(ELBO)
        return loss_list

    def replicate_data(self, lf_model, lf_guide, if_save=True):
        res = pred(lf_model, guide=lf_guide, num_samples=300)
        post_sample = res.forward(None)
        if if_save:
            to_pickle(post_sample['a'], "data_pickle/lf_post_sample_a")
            z_hat = post_sample['z'].mean(axis=0)
            to_pickle(z_hat, "data_pickle/lf_z_hat")
        return post_sample

    def plot_lf_ppc(self, lf_data, post_sample_a, is_cuda=True):
        true_len = lf_data.sum(axis=0).detach().cpu().numpy().astype(np.int)
        ppc_len = post_sample_a.sum(axis=1).mean(axis=0) \
                               .detach().cpu().numpy().astype(np.int)

        ind = np.arange(self.fea_num)
        tt = np.stack([ind, true_len]).T
        tm = np.stack([ind, ppc_len]).T

        tru = np.ones([self.fea_num, 1])
        mad = np.zeros([self.fea_num, 1])
        tt = np.concatenate([tt, tru], axis=1)
        tm = np.concatenate([tm, mad], axis=1)
        ti = np.concatenate([tt, tm], axis=0)
        df = pd.DataFrame(ti, columns=["feat_index", "count", "real/repl"])

        plt.subplots(figsize=(20, 6))
        sns.barplot(x="feat_index", y="count", hue="real/repl", data=df)
