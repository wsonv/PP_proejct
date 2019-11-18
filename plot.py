from models.NB import NeB
from models.Poisson import Pois
from models.mlrModel import MlrModel
from preprocessor import YelpData,load,to_pickle
import torch
import numpy as np
import criticism, inference
from inference import svi
from criticism import svi_posterior, mcmc_posterior, mlr_sampling
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO, TracePredictive, TraceEnum_ELBO
import pandas as pd
from torch.distributions import constraints
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocessor import YelpData, to_pickle
from pyro.infer.autoguide import init_to_feasible


def plot_loss(loss_list):
    ax = sns.lineplot(y=loss_list,x=range(1,len(loss_list)+1))

def plot_count_dist(samples,train_ratings,is_cuda = False):
    samples = np.squeeze(samples)
    sample_dict = process_ppc_data(samples, is_cuda)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    for i, ax in enumerate(axs.reshape(-1)):
        sns.distplot(sample_dict[i],ax=ax)
        ax.axvline(len([ k for k in train_ratings if k == i ]), 0,1,color='r')
        ax.set_title("{} star".format(i/2+1))
#     if mode == "save":
#         to_pickle(sample_dict,"{}_{}_sample_count_dict".format(model_type,infer_type))

def process_ppc_data(samples,is_cuda = False):
    sample_dict = dict()
    for r in range(9):
        sample_dict[r]=[]
    for i in range(9):
        temp = ((samples == i)*1).sum(axis = 1)
        if is_cuda:
            temp = temp.detach().cpu().numpy()
        else:
            temp = temp.detach().numpy()
        sample_dict[i] = temp
    # for i in range(samples.shape[0]):
    #     for j in range(9):
    #         sample_dict[j].append(len([ k for k in samples[i] if k == j]))
    return sample_dict
        

# def plot_svi_pred_dist(data,ratings,betas,svi_model,model,model_type):
#     rating_cat = [1,1.5,2,2.5,3,3.5,4,4.5,5]
#     plt.subplots(figsize=(15, 6))
#     color = sns.color_palette("Paired")
#     for i,s in enumerate(rating_cat):

#         d = data[ratings == s]
#         sample_data = svi_sampling(svi_model, d, [s]*d.shape[0], model, model_type, mode="nosave")
#         sample_data = np.squeeze(sample_data)
#         sample_data_ave = np.average(sample_data,axis=0)
#         sns.distplot(sample_data_ave,label="{}".format(s),rug=True, hist=False,kde_kws={"shade": True},color=color[i])

def plot_rating_dist(samples,train_ratings):
    # import pdb
    # pdb.set_trace()
    
    samples = np.squeeze(samples)
    rating_index_dict = dict()
    for r in range(9):
        rating_index_dict[r] = np.squeeze(np.argwhere(train_ratings==r))
        
    plt.subplots(figsize=(20, 6))
    color = sns.color_palette("Paired")
    for i in range(9):

        sample_data = samples[:,rating_index_dict[i]]
        sample_data_in_star = sample_data/2 + 1
        sample_data_ave = np.average(sample_data_in_star,axis=1)
        sns.distplot(sample_data_ave,label="{}".format(i/2+1),rug=True, hist=False,kde_kws={"shade":True},color=color[i])
        
def plot_beta_value(betas):
    res = np.zeros([1,3])
    count = 0
    for k,w in betas.items():
        for i in range(w.shape[1]):

            temp_1 = np.ones([w.shape[0],1]) * i
            temp_3 = np.ones([w.shape[0],1]) * count
            temp_2 = np.expand_dims(w[:,i],axis =1)
            tmp_res = np.concatenate([temp_1.astype(int),temp_2,temp_3.astype(int)],axis = 1)
            res = np.concatenate([res,tmp_res], axis = 0)
        count += 1
    res = res[1:]
    df_beta = pd.DataFrame(res, columns = ["category_index","beta_value","beta_index"])
    plt.subplots(figsize=(20, 6))
    ax = sns.lineplot(x="category_index", y="beta_value", hue="beta_index", data=df_beta, ci=None)
    

def plot_data_heatmap(rest_data,top_number=30):
    ###### Build up Heatmap ######
    
    cat_freq_dict = dict()
    for l in rest_data['categories']:
        if l != None:
            if 'Restaurants' in l and 'Food' in l:
                cats = [c for c in l.split(', ')]
                for c in cats:
                    if c in cat_freq_dict:
                        cat_freq_dict[c]+= 1
                    else:
                        cat_freq_dict[c] = 1

    cat_freq_tuples = [(cat_freq_dict[c],c) for c in cat_freq_dict]
    cat_freq_tuples.sort()
    cat_freq_tuples.reverse()

    n = top_number
    top_n = cat_freq_tuples[2:(2+n)]
    top_n = [t[1] for t in top_n]

    cat_star_dict = {c:[] for c in top_n}
    for i in range(len(rest_data)):
        sen = rest_data.iloc[i]
        cats = sen['categories'].split(", ")
        s = sen['stars']
        for top in cat_star_dict:
            if top in cats:
                cat_star_dict[top].append(s)


    cols = [1,1.5,2,2.5,3,3.5,4,4.5,5]
    index = top_n
    heatmap = np.zeros((len(top_n),len(cols)))
    for cat in cat_star_dict:
        i = top_n.index(cat)
        for r in cat_star_dict[cat]:
            col = cols.index(r)
            heatmap[i][col] += 1
            
    df_heatmap = pd.DataFrame(heatmap.T, index=cols, columns=index,dtype=int)
    f, ax = plt.subplots(figsize=(20, 9))
    plot_hm = sns.heatmap(df_heatmap, annot=True, fmt="d", ax=ax)


