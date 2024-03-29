import json
import pandas as pd
import numpy as np
import pickle


class YelpData:

    def __init__(self):
        self.rest_data = None
        self.categories = None
        self.ratings = None
        self.cat_one_hot = None
        self.index_length = None
        self.cities_to_include = ['Toronto', 'Las Vegas', 'Phoenix',
                                  'Charlotte', 'Calgary', 'Montréal',
                                  'Pittsburgh', 'Scottsdale', 'Cleveland',
                                  'Mesa']

    def process(self):
        # load json file to dataframe
        business = []
        for line in open('../yelp_dataset/business.json', 'r'):
            business.append(json.loads(line))
        bus_json = {"business": business}
        df = pd.DataFrame(bus_json['business'])

        cat_counts = {}
        for i in range(len(df)):
            sen = df.iloc[i]
            if sen['categories'] is None or \
               sen['city'] not in self.cities_to_include:
                continue
            elif 'Restaurants' in sen['categories'] and \
                 'Food' in sen['categories'] and \
                 sen['city'] in self.cities_to_include:
                cats = sen['categories'].split(", ")
                for word in cats:
                    if word in cat_counts:
                        cat_counts[word] += 1
                    else:
                        cat_counts[word] = 1

        ordered_cat_counts = [[c, w] for w, c in cat_counts.items()]
        ordered_cat_counts.sort()
        uncensored_cats_list = np.array(ordered_cat_counts)[:, 1]
        censoring = uncensored_cats_list[:228]
        censored = uncensored_cats_list[228:]
        censor_dict = {x: 0 for x in censoring}

        idxs = []
        cities = set()
        for i in range(len(df)):
            sen = df.iloc[i]
            if sen['categories'] is None or \
               sen['city'] not in self.cities_to_include:
                continue
            elif 'Restaurants' in sen['categories'] and \
                 'Food' in sen['categories'] and \
                 sen['city'] in self.cities_to_include:
                cats = sen['categories'].split(", ")
                is_pass = False
                for word in cats:
                    if word in censor_dict:
                        is_pass = True
                if not is_pass:
                    cities.add(sen['city'])
                    idxs.append(sen.name)
        rest_data = df.iloc[idxs]
        cat_list = list(cities) + censored.tolist()
        cat_list.remove('Restaurants')
        cat_list.remove('Food')
        self.index_length = len(cat_list)

        word_to_index = {w: i for i, w in enumerate(cat_list)}
        # one hot encoder
        cat_one_hot = np.zeros((len(rest_data), self.index_length))
        for i in range(len(rest_data)):
            put_city = False
            line = rest_data.iloc[i]
            c_col = line['categories']
            if c_col is not None:
                cats = c_col.split(', ')
                for c in cats:
                    if c != 'Restaurants' and c != 'Food':
                        cat_one_hot[i][word_to_index[c]] = 1
                        put_city = True
                if put_city:
                    cat_one_hot[i][word_to_index[line['city']]] = 1

        self.cat_counts = cat_counts
        self.rest_data = rest_data
        self.categories = cat_list
        self.ratings = rest_data['stars']
        self.cat_one_hot = cat_one_hot

    def add_bias(self):
        bias = np.ones([self.cat_one_hot.shape[0], 1])
        cat_one_hot_bias = np.concatenate([self.cat_one_hot, bias], axis=1)
        return cat_one_hot_bias

    def get_vector(self, cats, index_list=None):
        if index_list:
            res = np.zeros([1, len(index_list) + 1])
            for w in cats:
                res[0][index_list[w]] = 1
        else:
            res = np.zeros([1, self.index_length + 1])
            for w in cats:
                res[0][self.cat_index[w]] = 1
        # adding bias
        res[0][-1] = 1
        return res


def to_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)


def load(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data
