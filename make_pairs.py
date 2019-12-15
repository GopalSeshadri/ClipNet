"""
Created on Mon Dec 05 19:07:02 2019

@author: Gopal Seshadri
"""
import numpy as np
import pandas as pd
import pickle
import json

# Loading frame and caption embeddings
def load_embedding(file_name, embedding_type):
    '''
    This function takes in a file name and embedding type, whether it is a frame or caption.
    Return a dictionary of that embedding type.

    Parameters:
    file_name (str) : A file name of the embedding
    embedding_type (str) : It can either of two values, 'frame' or 'caption'

    Returns:
    dict : A dictionary of embedding with video id as key
    '''

    if embedding_type == 'frame':
        with open('Embeddings/frame_embeddings/{}'.format(file_name), 'rb') as f:
            return pickle.load(f)
    elif embedding_type == 'caption':
        with open('Embeddings/caption_embeddings/{}'.format(file_name), 'rb') as f:
            return json.load(f)
    else:
        return

def create_pairs(frame_embeddings, caption_embeddings):
    '''
    This function takes in frame embeddings and caption emebeddding as dictionary. Returns a list of paired data.

    Parameters:
    frame_embeddings (dict) : A dictionary of frame embeddings
    caption_embeddings (dict) : A dictionary of caption embeddings

    Returns:
    output_list (list) : A list of lists, where each list of structure,
                         [clip_sentence, clip_frames, clip_captions, video id, timestamp]
    '''
    output_list = []

    for i, video_id in enumerate(caption_embeddings.keys()):
        vid = video_id[2:]

        sentences = caption_embeddings[video_id]["sentences"]
        duration = caption_embeddings[video_id]["duration"]
        timestamps = np.array(caption_embeddings[video_id]["timestamps"])
        vectors = caption_embeddings[video_id]["vectors"]

        frames = frame_embeddings.get(vid + '.mp4', None)
        if frames is None:
            continue

        idx = np.rint((frames.shape[0] * timestamps) / duration).astype(int)

        for j in range(len(sentences)):
            clip_frames = frames[np.arange(idx[j, 0], idx[j, 1]), :]
            clip_captions = np.array(vectors[j])
            clip_timestamp = timestamps[j]
            clip_sentence = sentences[j]
            if clip_frames.shape[0] < 3:
                continue
            output_list.append([clip_sentence, clip_frames, clip_captions, vid, clip_timestamp])

    # print(output_list)
    return output_list

#### Note: ####
## The key values of frame embedding has video id with .mp4 suffix
## The key values of caption embeddings has video id with v_ prefix

train_frame_embedding = load_embedding('feature_dict.pkl', 'frame') # file name might change
train_caption_embeddings = load_embedding('train_embeddings.json', 'caption')

train_pair_list = create_pairs(train_frame_embedding, train_caption_embeddings)
print(len(train_pair_list))

with open('Paired/train_pair.pickle', 'wb') as f:
    pickle.dump(train_pair_list, f, protocol = pickle.HIGHEST_PROTOCOL)

ids = [each[3] for each in train_pair_list]
print(ids)
print(len(set(ids)))
