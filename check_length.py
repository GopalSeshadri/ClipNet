# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:22:15 2019

@author: vjkri
"""

print('Importing all required libraries....')
import os
import cv2

#path to directories
temp_dir = 'Data/Original'
save_dir = 'Data/Reduced'

temp_vids = os.listdir(temp_dir)

for vid in temp_vids:

    file = os.path.join(temp_dir, vid)
    red_file = os.path.join(save_dir, vid)

    #read both original video and resolution reduced video files
    cap = cv2.VideoCapture(file)
    cap_ = cv2.VideoCapture(red_file)

    #check if both videos has the same frames per second
    if cap.get(cv2.CAP_PROP_POS_MSEC) != cap_.get(cv2.CAP_PROP_POS_MSEC):
        print(vid)
        print(False)
