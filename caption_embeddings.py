"""
@author: Agastya Teja Anumanchi
"""

import numpy as np
import json


def remove_mp4(id_list):
    '''
    Helper function for removing .mp4 extention

    Parameters :
    id_list (list) : Takes in ids with .mp4 extention

    Returns :
    id_list (list): Returns a list with .mp4 removed
    '''

    for i in range(len(id_list)):
        id_list[i] = id_list[i][:-4]

    return id_list


def load_json(path):
    '''
    Helper function to load json file

    '''

    with open(path,"r") as f:

        '''
            Json structure: datatype
            Videoid: String,
            Duration: Numeric,
            Timestamps: List,
            Sentences: String

        '''

        videos = json.load(f)

    return videos

def intersection_ids(all_keys,temp,ids):
        '''
        Helper function that gives intersection between chosen classes and all caption dump

        Parameters :
        all_keys (list) : all id caption worddump
        temp(list): list of id's without "V_"
        ids(list): list of ids's with "V_"

        Returns :
        temp_output (list): Returns a list of intersection ids between caption and videos
        '''

    temp_output = []
    count = 0

    for id in ids:

        if id in temp:

            index = temp.index(id)
            temp_output.append(all_keys[index])
            count+=1

    return temp_output

def create_sentence_embeddings(vectors):

    '''
    Helper function that creats embedding to the entire sentence by taking average for entire sentence

    Parameters :
    vectors(list): Takes in a list of 300 dim word embeddings for entire sentence

    Returns :
    sentence_embeddings (list): Returns a 300 dim sentence embedding
    ''''


    sentence_embeddings = []
    for sentence in vectors:
        #print(len(sentence))
        #print(len(sentence), len(sentence[0]))
        sentence_array = np.array(sentence)
        #print(sentence_array.shape)
        sentence_array_reduced = np.mean(sentence_array, axis = 0)
        print(sentence_array_reduced.shape)

        sentence_embeddings.append(sentence_array_reduced.tolist())
    return sentence_embeddings


def word_embeddings(id_list):

    '''
    Helper function that takes in intersection id list and subsets these ids from the master dump

    Parameters :
    id_list(list): List of intersection ids

    Returns :
    final_output (dict): Returns a dict based on the intersection ids
    '''
    final_output = {}

    xyz = []
    for id in id_list:

        if id in train_word_embeddings.keys():
            final_output[id] = train_word_embeddings[id]

        elif id in val1_word_embeddings.keys():
            final_output[id] = val1_word_embeddings[id]

        elif id in val2_word_embeddings.keys():
            final_output[id] = val2_word_embeddings[id]

    return final_output


video_id_list_valid = np.load('Data/ChosenClasses/video_id_list_valid.pkl', allow_pickle=True)
video_id_list_train = np.load('Data/ChosenClasses/video_id_list_train.pkl', allow_pickle=True)

val_id_list = remove_mp4(video_id_list_valid)
train_id_list = remove_mp4(video_id_list_train)

train_word_embeddings = load_json(path = "Data/Captions/worddump/all_train.json")
val1_word_embeddings = load_json(path = "Data/Captions/worddump/val_1.json")
val2_word_embeddings = load_json(path = "Data/Captions/worddump/val_2.json")
all_keys = list(train_word_embeddings.keys()) + list(val1_word_embeddings.keys()) + list(val2_word_embeddings.keys())
temp = [i[2:] for i in all_keys]

train_id_intersection = intersection_ids(all_keys,temp,ids=train_id_list)
validation_id_intersection = intersection_ids(all_keys,temp,ids=val_id_list)

train_embeddings = word_embeddings(id_list=train_id_intersection)
val_embeddings = word_embeddings(id_list=validation_id_intersection)

for key in train_embeddings.keys():
    train_embeddings[key]['vectors'] = create_sentence_embeddings(train_embeddings[key]['vectors'])

with open("Embeddings/caption_embeddings/train_embeddings.json", "w") as f:
    json.dump(train_embeddings, f)

for key in val_embeddings.keys():
    val_embeddings[key]['vectors'] = create_sentence_embeddings(val_embeddings[key]['vectors'])

with open("Embeddings/caption_embeddings/val_embeddings.json", "w") as f:
    json.dump(val_embeddings, f)
