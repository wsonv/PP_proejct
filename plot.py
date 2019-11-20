import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss(loss_list):
    sns.lineplot(y=loss_list, x=range(1, len(loss_list)+1))


def plot_count_dist(samples, train_ratings, is_cuda=False):
    samples = np.squeeze(samples)
    sample_dict = process_ppc_data(samples, is_cuda)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    for i, ax in enumerate(axs.reshape(-1)):
        sns.distplot(sample_dict[i], ax=ax)
        ax.axvline(len([k for k in train_ratings if k == i]), 0, 1, color='r')
        ax.set_title("{} star".format(i/2+1))


def process_ppc_data(samples, is_cuda=False):
    sample_dict = dict()
    for r in range(9):
        sample_dict[r] = []
    for i in range(9):
        temp = ((samples == i) * 1).sum(axis=1)
        if is_cuda:
            temp = temp.detach().cpu().numpy()
        else:
            temp = temp.detach().numpy()
        sample_dict[i] = temp
    return sample_dict


def plot_rating_dist(samples, train_ratings):
    samples = np.squeeze(samples)
    rating_index_dict = dict()
    for r in range(9):
        rating_index_dict[r] = np.squeeze(np.argwhere(train_ratings == r))

    plt.subplots(figsize=(20, 6))
    color = sns.color_palette("Paired")
    for i in range(9):

        sample_data = samples[:, rating_index_dict[i]]
        sample_data_in_star = sample_data/2 + 1
        sample_data_ave = np.average(sample_data_in_star, axis=1)
        sns.distplot(sample_data_ave, label="{}".format(i/2+1), rug=True,
                     hist=False, kde_kws={"shade": True}, color=color[i])


def plot_beta_value(betas):
    res = np.zeros([1, 3])
    count = 0
    for k, w in betas.items():
        for i in range(w.shape[1]):

            temp_1 = np.ones([w.shape[0], 1]) * i
            temp_3 = np.ones([w.shape[0], 1]) * count
            temp_2 = np.expand_dims(w[:, i], axis=1)
            tmp_res = np.concatenate([temp_1.astype(int), temp_2,
                                      temp_3.astype(int)], axis=1)
            res = np.concatenate([res, tmp_res], axis=0)
        count += 1
    res = res[1:]
    df_beta = pd.DataFrame(res, columns=["category_index",
                                         "beta_value",
                                         "beta_index"])
    plt.subplots(figsize=(20, 6))
    sns.lineplot(x="category_index", y="beta_value", hue="beta_index",
                 data=df_beta, ci=None, legend='full')


def plot_data_heatmap(rest_data, top_number=30):
    cat_freq_dict = dict()
    for l in rest_data['categories']:
        if l is not None:
            if 'Restaurants' in l and 'Food' in l:
                cats = [c for c in l.split(', ')]
                for c in cats:
                    if c in cat_freq_dict:
                        cat_freq_dict[c] += 1
                    else:
                        cat_freq_dict[c] = 1

    cat_freq_tuples = [(cat_freq_dict[c], c) for c in cat_freq_dict]
    cat_freq_tuples.sort()
    cat_freq_tuples.reverse()

    n = top_number
    top_n = cat_freq_tuples[2:(2+n)]
    top_n = [t[1] for t in top_n]

    cat_star_dict = {c: [] for c in top_n}
    for i in range(len(rest_data)):
        sen = rest_data.iloc[i]
        cats = sen['categories'].split(", ")
        s = sen['stars']
        for top in cat_star_dict:
            if top in cats:
                cat_star_dict[top].append(s)

    cols = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    index = top_n
    heatmap = np.zeros((len(top_n), len(cols)))
    for cat in cat_star_dict:
        i = top_n.index(cat)
        for r in cat_star_dict[cat]:
            col = cols.index(r)
            heatmap[i][col] += 1

    df_heatmap = pd.DataFrame(heatmap.T, index=cols, columns=index, dtype=int)
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(df_heatmap, annot=True, fmt="d", ax=ax)


def plot_compared_beta_value(svi_betas, mcmc_betas):

    fig, axs = plt.subplots(nrows=9, ncols=1, figsize=(20, 60))
    beta_list = ['beta_1', 'beta_1h', 'beta_2', 'beta_2h', 'beta_3',
                 'beta_3h', 'beta_4', 'beta_4h', 'beta_5']
    for i, ax in enumerate(axs.reshape(-1)):

        res = np.zeros([1, 3])
        k = beta_list[i]
        w_1 = svi_betas[k]
        w_2 = mcmc_betas[k]
        for i in range(w_1.shape[1]):

            temp_1 = np.ones([w_1.shape[0], 1]) * i
            temp_3 = np.ones([w_1.shape[0], 1]) * 0
            temp_2 = np.expand_dims(w_1[:, i], axis=1)
            tmp_res = np.concatenate([temp_1.astype(int),
                                      temp_2, temp_3], axis=1)
            res = np.concatenate([res, tmp_res], axis=0)
            temp_4 = np.ones([w_2.shape[0], 1]) * i
            temp_6 = np.ones([w_2.shape[0], 1]) * 1
            temp_5 = np.expand_dims(w_2[:, i], axis=1)
            tmp_res = np.concatenate([temp_4.astype(int),
                                      temp_5, temp_6], axis=1)
            res = np.concatenate([res, tmp_res], axis=0)
        res = res[1:]
        df_beta = pd.DataFrame(res, columns=["category_index",
                                             "beta_value", "svi/mcmc"])

        sns.lineplot(x="category_index", y="beta_value", hue="svi/mcmc",
                     data=df_beta, ci=None, ax=ax, legend="full")
        ax.set_title("Beta for {} star".format(i/2+1))
