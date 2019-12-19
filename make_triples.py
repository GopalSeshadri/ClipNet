import numpy as np
import pandas as pd
import pickle
import random

def pad_pairs(pair_list):
    '''
    Takes in a list of pairs returns a list of tuples

    Parameters:
    pair_list (list) : A list of lists, where each list of structure,
                      [clip_sentence, clip_frames, clip_captions, video id, timestamp]

    Returns:
    pair_list (list) : A list of lists, where each list of structure
    '''
    # max_frame_step = max([each[1].shape[0] for each in pair_list])
    max_frame_step = 50
    # print(max_frame_step)
    for idx, each in enumerate(pair_list):
        pair_list[idx][1] = np.vstack([each[1], np.zeros((max_frame_step - each[1].shape[0], 500))])

    print([each[1].shape for each in pair_list])
    # print([t.shape for t in temp])

    # max_caption_step = max([each[2].shape[0] for each in pair_list])
    max_caption_step = 50
    # print(max_caption_step)

    for idx, each in enumerate(pair_list):
        if each[2].shape[0] < 50:
            pair_list[idx][2] = np.vstack([each[2], np.zeros((max_caption_step - each[2].shape[0], 300))])
        else:
            pair_list[idx][2] = each[2][:50, :]

    print([each[2].shape for each in pair_list])
    return pair_list

def create_triples(pair_list):
    '''
    Takes in a list of pairs returns a list of triplets

    Parameters:
    pair_list (list) : A list of lists, where each list of structure,
                      [clip_sentence, clip_frames, clip_captions, video id, timestamp]

    Returns:
    triple_list (list) : A list of triplets [anchor, positive, negative]
    '''
    
    triple_list = []
    for i, each in enumerate(pair_list):
        anchor = each[2]
        positive = each[1]
        temp = pair_list[: i] + pair_list[i + 1 :]
        indices = random.sample(range(0, len(temp)), 250)
        for j in indices:
            negative = temp[j][1]
            triple_list.append([anchor, positive, negative])

    return triple_list


# Creating triples for training set
with open('Paired/train_pair.pickle', 'rb') as f:
    train_pair_list = pickle.load(f)

train_pair_list = pad_pairs(train_pair_list)

train_triples_list = create_triples(train_pair_list)
print(len(train_triples_list))

with open('Paired/train_triples_list.pickle', 'wb') as f:
    pickle.dump(train_triples_list, f, protocol = pickle.HIGHEST_PROTOCOL)

# Creating triples for validation set
with open('Paired/val_pair.pickle', 'rb') as f:
    val_pair_list = pickle.load(f)

val_pair_list = pad_pairs(val_pair_list)

val_triples_list = create_triples(val_pair_list)
print(len(val_triples_list))

with open('Paired/val_triples_list.pickle', 'wb') as f:
    pickle.dump(val_triples_list, f, protocol = pickle.HIGHEST_PROTOCOL)
