# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:36:47 2019

@author: vjkri
"""

import os
import numpy as np
from sklearn.decomposition import PCA
import pickle

#reduce to 500 dimensions from 4096
pca = PCA(n_components = 500)

# working directory
work_dir = ''

with open(os.path.join(work_dir, 'painting.pkl'), 'rb') as f:
    features = pickle.load(f)

with open(os.path.join(work_dir, 'info_dict.pkl'), 'rb') as f:
    dict_info = pickle.load(f)

with open('Using_uneven_bars.pkl', 'rb') as f:
    last_8_features = pickle.load(f)

with open('info_dict_1.pkl', 'rb') as f:
    info_dict_1 = pickle.load(f)

# concatenate info_dict_1 with dict_info
for key, values in info_dict_1.items():
    dict_info.update({key : values})

# concatenate 12 classes with last 8 classes
# all_features is a list with all 20 classes features
all_features = features + last_8_features


all_features = np.array(all_features)
all_features = np.reshape(all_features, (all_features.shape[0], all_features.shape[2]))

# apply pca to all_features array
all_features_pca = pca.fit_transform(all_features)

# map features with the corresponding video ids
num_total_count = 0
feature_dict = {}
for key, values in dict_info.items():
    print(key)
    #print(values)
    for key, value in values.items():
        #print(features[num_total_count: num_total_count + value])
        feature_dict[key] = all_features_pca[num_total_count: num_total_count + value]
        num_total_count += value

# dump feature dictionary
with open('feature_dict.pkl', 'wb') as f:
    pickle.dump(feature_dict, f)
