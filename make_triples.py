import numpy as np
import pandas as pd
import pickle
import random


def create_triples(pair_list):
    '''
    Takes in a list of pairs returns a list of tuples

    Parameters:
    pair_list (list) : A list of lists, where each list of structure,
                      [clip_sentence, clip_frames, clip_captions, video id, timestamp]

    Returns:

    '''
    triple_list = []
    for i, each in enumerate(pair_list):
        anchor = each[2]
        positive = each[1]
        temp = pair_list[: i] + pair_list[i + 1 :]
        indices = random.sample(range(0, len(temp)), 200)
        for j in indices:
            negative = temp[j][1]
            triple_list.append([anchor, positive, negative])

    return triple_list


# Creating triples for training set
with open('Paired/train_pair.pickle', 'rb') as f:
    train_pair_list = pickle.load(f)

train_triples_list = create_triples(train_pair_list)
print(len(train_triples_list))

with open('Paired/train_triples_list.pickle', 'wb') as f:
    pickle.dump(train_triples_list, f, protocol = pickle.HIGHEST_PROTOCOL)

# Creating triples for validation set
with open('Paired/val_pair.pickle', 'rb') as f:
    val_pair_list = pickle.load(f)

val_triples_list = create_triples(val_pair_list)
print(len(val_triples_list))

with open('Paired/val_triples_list.pickle', 'wb') as f:
    pickle.dump(val_triples_list, f, protocol = pickle.HIGHEST_PROTOCOL)
