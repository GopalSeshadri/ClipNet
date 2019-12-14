# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:38:08 2019

@author: vjkri
"""

print('Importing all required libraries....')

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import cv2
import os
import pickle

print('Finished importing libraries')

#importing VGG16 model from Keras
model = VGG16(weights='imagenet', include_top = True)

#removing output layer for feature extraction
vgg16_model = tf.keras.models.Sequential()

for layer in model.layers[:-1]:
    vgg16_model.add(layer)

print('Created VGG16 model')

#creating reference dictionary
info_dict = {}

#main directory
main_dir_train = 'Data/Reduced/Videostraining'
main_dir_valid = 'Data/Reduced/Videosvalidation'

#list of class directories
main_subdir_train = os.listdir(main_dir_train)
main_subdir_valid = os.listdir(main_dir_valid)


#empty list to store extracted features
all_vid_list = []

print('Starting feature extraction')

with open('chosen_classes.pkl', 'rb') as f:
    chosen_classes = pickle.load(f)

print(chosen_classes)

for subdir in main_subdir_train:
    #create a new dictionary with class name
    if subdir in chosen_classes:
        info_dict[subdir] = {}

        temp_dir = os.listdir(os.path.join(main_dir_train, subdir))

        print('Extracting features for the class ' + str(subdir) + ' ............')

        for vid_name in temp_dir:
            print(vid_name)
            file = os.path.join(main_dir_train, subdir, vid_name)
            video = cv2.VideoCapture(file)

            count = 1
            total_fr_in_vid = 0

            while True:
                #reading a frame in video
                ret, frame = video.read()

                if ret == True:
                    #sampling frame with 16 frame intervals
                    if count % 16 == 0:
                        #resizing frame size to feed into VGG16
                        frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

                        #converting image to array
                        x = image.img_to_array(frame)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)

                        #extracting features using fc7 of VGG16
                        features = vgg16_model.predict(x)

                        #appending extracted features to a list
                        all_vid_list.append(features)

                        total_fr_in_vid += 1
                    count += 1
                else:
                    break

        #recording number of frames in a particular video
            info_dict[subdir][vid_name] = total_fr_in_vid
    print('Extracted features for the class ' + str(subdir))

    with open(str(subdir) + '.pkl', 'wb') as f:
        pickle.dump(all_vid_list, f)

    with open('info_dict.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

print('Pickle all files')

#pickle features extracted
with open('all_vid_list.pkl', 'wb') as f:
    pickle.dump(all_vid_list, f)

#pickle reference dictionary
with open('info_dict.pkl', 'wb') as f:
    pickle.dump(info_dict, f)

print('Execution successful')

with open('info_dict.pkl', 'rb') as f:
    temps = pickle.load(f)

print(len(temps))
