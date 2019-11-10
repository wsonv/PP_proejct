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
        

    def process(self):
        # load json file to dataframe
        business = []
        for line in open('./yelp_dataset/business.json', 'r'):
            business.append(json.loads(line))
        bus_json = {"business":business}
        df = pd.DataFrame(bus_json['business'])

        # extract restaurants
        number = set()
        idxs = []
        for i in range(len(df)):
            sen = df.iloc[i]
            if sen['categories'] is None:
                continue
            elif 'Restaurants' in sen['categories'] and 'Food' in sen['categories']:
                number.update(sen['categories'].split(", "))
                idxs.append(sen.name)
        rest_data = df.iloc[idxs]
        cat_list = list(number)
        cat_list.remove('Restaurants')
        cat_list.remove('Food')
        
        # one hot encoder
        cat_one_hot = np.zeros((len(rest_data),len(cat_list)))
        for i in range(len(rest_data)):
            l = rest_data.iloc[i]
            c_col = l['categories']
            if c_col != None:
                cats = [c for c in c_col.split(', ')]
                if "Handyman" in cats:
                    import pdb
                    pdb.set_trace()
                for c in cats:
                    if c != 'Restaurants' and c != 'Food':
                        cat_one_hot[i][cat_list.index(c)] = 1

        self.rest_data = rest_data
        self.categories = cat_list
        self.ratings = rest_data['stars']
        self.cat_one_hot = cat_one_hot
        
    def add_bias(self):
        cat_one_hot_bias = np.ones((self.cat_one_hot.shape[0],self.cat_one_hot.shape[1]+1))
        for i in range(len(cat_one_hot_bias)):
            encoded_cats = self.cat_one_hot[i]
            cat_one_hot_bias[i][:len(encoded_cats)] = encoded_cats
        return cat_one_hot_bias
    
    def to_pickle(self, data, name):
        with open(name, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, name):
        with open(name, "rb") as f:
            data = pickle.load(f)
        return data