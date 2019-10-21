import populartimes
import pickle


data = populartimes.get("AIzaSyCOj7mcZ-ac2Mr7XYSk-ZOub-q_QeyOgLQ",["restaurant"], (40.701343,-74.015773), (40.73301,-73.973981), n_threads=1, max_num = 1000, max_district_num = 100)

with open('crawled_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('crawled_data.pickle', 'rb') as handle:
#     b = pickle.load(handle)
