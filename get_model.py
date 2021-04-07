# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:14:38 2021

@author: dev
"""

from tensorflow.keras.applications.vgg16 import VGG16

model=VGG16()

model.save('mymodel.h5')
print("VGG16 model downloaded and saved successfully")