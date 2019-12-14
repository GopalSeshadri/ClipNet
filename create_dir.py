# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:30:44 2019

@author: vjkri
"""

import os

main_dir = 'Data/Original'
reduce_main_dir = 'Data/Reduced'


#Below piece of code should only be run once, only after deleting the reduced_main_dir completely
#it should be run again

#creating 'Data/Reduced' directory
os.mkdir(reduce_main_dir)

train_valid_test = os.listdir(main_dir)

for direc in train_valid_test:
    os.mkdir(os.path.join(reduce_main_dir, direc))

for direc in train_valid_test:
    sub_direcs = os.listdir(os.path.join(main_dir, direc))

    for sub_dir in sub_direcs:
        os.mkdir(os.path.join(reduce_main_dir, direc, sub_dir))
