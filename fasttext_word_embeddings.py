import gensim
import numpy as np
import pandas as pd
import json
import string
from gensim.models import KeyedVectors as KV

# This model file is downloaded from the fasttext webpage: https://fasttext.cc/docs/en/english-vectors.html
model = KV.load_word2vec_format('Embeddings/Fasttext/wiki-news-300d-1M.vec', binary=False)
print('Model Loaded')

def text_preprocess(each_text):
    '''
    Helper function for preprocessing the text. Removes punctuations and digits in a text

    Parameters :
    each_text (string) : Takes in word in a sentence

    Returns :
    output (string): Returns a string
    '''

    punctuations_digits = string.punctuation + string.digits
    output = each_text.translate(str.maketrans('', '', punctuations_digits)

    return output

def clean_punctuation(text):
    '''
    Helper function for removing punctuations in setence. Replaces punctuations with spaces.

    Parameters :
    text (string): Takes in a sentence

    Returns:
    processed (string) : Returns a string
    '''

    string_punctuations = string.punctuation
    processed = (text.lower()).translate(str.maketrans(string_punctuations, ' ' * len(string.punctuation)))

    return processed

def load_json(path):
    '''
    Helper function to load json file

    Parameters :
    path (string): Takes in location of json file

    Returns:
    vidoes (dict) : Returns the loaded json file
    '''

    with open(path,"r") as f:
        '''
            Json structure: datatype
            {
                Videoid: String,
                Duration: Numeric,
                Timestamps: List,
                Sentences: String

            }
        '''

        videos = json.load(f)

    return videos

def sentence_to_vector(sentence):
    '''
    Helper function that takes each caption setence and returns 300 Dim vector

    Parameters :
    sentence (string): Takes in each caption as string

    Returns:
    vector_sentence (list) : Returns the vectorized sentece of 300 Dim in the form of list
    '''

    vector_sentence = []

    for each_word in sentence:

        cleaned_word = text_preprocess(each_word)

        if cleaned_word in model:

            vector_sentence.append(model[cleaned_word].tolist())


    return vector_sentence


def gen_caption_vec(captions):
    '''
    Helper function that takes each caption setence and returns 300 Dim vector

    Parameters :
    sentence (string): Takes in each caption as string

    Returns:
    vector_sentence (list) : Returns the vectorized sentece of 300 Dim in the form of list
    '''

    vector_caption = []

    for each_caption in captions:

        processed_caption = clean_punctuation(each_caption)

        processed_caption = processed_caption.strip()
        sentence = processed_caption.split()

        vector_caption.append(sentence_to_vector(sentence))

    return vector_caption

def write_output(videos,path):
    '''
    Helper function to write the output to json files

    Parameters:
    videos (dict),path (string) : Takes in videos dict and path

    Returns:
    None: We write the output in a particular location
    '''

    all_keys = list(videos.keys())
    i = 0
    lb = i*1000
    ub = (i+1)*1000

    while(i<10):
        print(i)

        keys = all_keys[lb:ub]
        video_subset = {}

        for key,value in videos.items():

            if key in keys:
                video_subset[key] = value


        with open(path + str(i+1) + ".json", "w") as f:

            json.dump(video_subset, f)

        i = i + 1


videos = load_json(path = "Data/Captions/train.json")
for id in videos:

    video = videos[id]
    captions = video['sentences']
    caption_vec = gen_caption_vec(captions)
    video['vectors'] = caption_vec

print('Training Caption Embeddings Complete')

with open("Data/Captions/train.json", "w") as f:

    json.dump(videos, f)

videos = load_json(path = "Data/Captions/val_1.json")
for id in videos:

    video = videos[id]
    captions = video['sentences']
    caption_vec = gen_caption_vec(captions)
    video['vectors'] = caption_vec

print('Validation 1 Caption Embeddings Complete')

with open("Data/Captions/val_1.json", "w") as f:

    json.dump(videos, f)

videos = load_json(path = "Data/Captions/val_2.json")
for id in videos:

    video = videos[id]
    captions = video['sentences']
    caption_vec = gen_caption_vec(captions)
    video['vectors'] = caption_vec

print('Validation 2 Caption Embeddings Complete')

with open("Data/Captions/val_2.json", "w") as f:

    json.dump(videos, f)
