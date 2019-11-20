import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints


class MlrModel:

    def model(self, data, ratings):
        mu = torch.zeros(9)
        sigma = torch.ones(9)
        with pyro.plate("betas", data.shape[1]):

            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))
            beta_5 = pyro.sample("beta_5", dist.Normal(mu[8], sigma[8]))
        p_1 = torch.sum(beta_1 * data, axis=1)
        p_1h = torch.sum(beta_1h * data, axis=1)
        p_2 = torch.sum(beta_2 * data, axis=1)
        p_2h = torch.sum(beta_2h * data, axis=1)
        p_3 = torch.sum(beta_3 * data, axis=1)
        p_3h = torch.sum(beta_3h * data, axis=1)
        p_4 = torch.sum(beta_4 * data, axis=1)
        p_4h = torch.sum(beta_4h * data, axis=1)
        p_5 = torch.sum(beta_5 * data, axis=1)

        p_array = torch.stack([p_1, p_1h, p_2, p_2h, p_3,
                               p_3h, p_4, p_4h, p_5])
        exp_sum = torch.sum(torch.exp(p_array), axis=0)

        softmax_array = torch.exp(p_array) / exp_sum
        # temp_total = torch.sum(softmax_array, axis=0)
        softmax_array = softmax_array.T

        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Categorical(probs=softmax_array),
                            obs=ratings)
        return y

    def emp_model(self, data, ratings):
        mu = torch.rand(9, data.shape[1]) * 4 - 2
        sigma = torch.ones(9)
        with pyro.plate("betas", data.shape[1]):

            beta_1 = pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
            beta_1h = pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
            beta_2 = pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
            beta_2h = pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
            beta_3 = pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
            beta_3h = pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
            beta_4 = pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
            beta_4h = pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))
            beta_5 = pyro.sample("beta_5", dist.Normal(mu[8], sigma[8]))
        p_1 = torch.sum(beta_1 * data, axis=1)
        p_1h = torch.sum(beta_1h * data, axis=1)
        p_2 = torch.sum(beta_2 * data, axis=1)
        p_2h = torch.sum(beta_2h * data, axis=1)
        p_3 = torch.sum(beta_3 * data, axis=1)
        p_3h = torch.sum(beta_3h * data, axis=1)
        p_4 = torch.sum(beta_4 * data, axis=1)
        p_4h = torch.sum(beta_4h * data, axis=1)
        p_5 = torch.sum(beta_5 * data, axis=1)

        p_array = torch.stack([p_1, p_1h, p_2, p_2h, p_3,
                               p_3h, p_4, p_4h, p_5])
        exp_sum = torch.sum(torch.exp(p_array), axis=0)

        softmax_array = torch.exp(p_array) / exp_sum
        # temp_total = torch.sum(softmax_array, axis=0)
        softmax_array = softmax_array.T

        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Categorical(probs=softmax_array),
                            obs=ratings)
        return y

    def guide(self, data, ratings):

        mu = pyro.param("mu", torch.zeros(9, data.shape[1]))
        sigma = pyro.param('sigma', torch.ones(9, data.shape[1]),
                           constraint=constraints.positive)

        with pyro.plate("betas", data.shape[1]):

            pyro.sample("beta_1", dist.Normal(mu[0], sigma[0]))
            pyro.sample("beta_1h", dist.Normal(mu[1], sigma[1]))
            pyro.sample("beta_2", dist.Normal(mu[2], sigma[2]))
            pyro.sample("beta_2h", dist.Normal(mu[3], sigma[3]))
            pyro.sample("beta_3", dist.Normal(mu[4], sigma[4]))
            pyro.sample("beta_3h", dist.Normal(mu[5], sigma[5]))
            pyro.sample("beta_4", dist.Normal(mu[6], sigma[6]))
            pyro.sample("beta_4h", dist.Normal(mu[7], sigma[7]))
            pyro.sample("beta_5", dist.Normal(mu[8], sigma[8]))
