import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints


class Pois:

    def model(self, data, ratings):

        with pyro.plate("betas", data.shape[1]):
            betas = pyro.sample("beta", dist.Normal(0, 1))

        lambda_ = torch.exp(torch.sum(betas * data, axis=1))
        with pyro.plate("ratings", data.shape[0]):
            y = pyro.sample("obs", dist.Poisson(lambda_), obs=ratings)

        return y

    def guide(self, data, ratings):
        locs = pyro.param('weights_loc', torch.ones(data.shape[1]))
        scales = pyro.param('weights_scale', torch.ones(data.shape[1]),
                            constraint=constraints.positive)

        with pyro.plate("betas", data.shape[1]):
            pyro.sample("beta", dist.Normal(locs, scales))

    def process_ratings(self, ratings):
        return ratings * 2 - 1

    def unprocess_ratings(self, ratings):
        return (ratings + 1) / 2
