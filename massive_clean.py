# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:00:08 2019

@author: vjkri
"""

print('Importing all required libraries....')
import cv2
import numpy as np
import os

#create origianl directory path for training and validation files
main_dir_train = 'Data/Original/Videostraining'
main_dir_valid = 'Data/Original/Videosvalidation'

#create directory path to store resolution reduced video files
reduce_dir_train = 'Data/Reduced/Videostraining'
reduce_dir_valid = 'Data/Reduced/Videosvalidation'

main_subdir_train = os.listdir(main_dir_train)
main_subdir_valid = os.listdir(main_dir_valid)

reduce_subdir_train = os.listdir(reduce_dir_train)
reduce_subdir_valid = os.listdir(reduce_dir_valid)

for subdir in main_subdir_valid:
    temp_dir = os.listdir(os.path.join(main_dir_valid, subdir))

    for vid_name in temp_dir:
        file = os.path.join(main_dir_valid, subdir, vid_name)
        save_file = os.path.join(reduce_dir_valid, subdir, vid_name)

        #read video file
        cap = cv2.VideoCapture(file)

        #read the frames per second of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'XVID') , fps, (140,80))

        #empty list to store all resized frames in the current video
        img_array = []

        while True:
            #read the frame (ret will be True if there is a frame and will be False if there is no frame)
            ret, frame = cap.read()
            if ret == True:
                #resize the read frame
                b = cv2.resize(frame, (140,80), fx=0, fy=0, interpolation = cv2.INTER_AREA)
                #append resized frame to the list
                img_array.append(b)
            else:
                break

        #write the video with the resized frames
        for i in range(len(img_array)):
            out.write(img_array[i])

        cap.release()
        out.release()
        cv2.destroyAllWindows()
