# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:25:37 2021

@author: dev
"""

import os
import numpy as np
#import PIL
import tensorflow 
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input , decode_predictions
from tensorflow.keras.models import load_model



model=tensorflow.keras.models.load_model('mymodel.h5')
print("model is loaded\n")

imgpath='imageset/images1.jpg'
img=load_img(imgpath,target_size=(224,224))
img=img_to_array(img)
img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
img=preprocess_input(img)

predictions=model.predict(img)
prediction_result=decode_predictions(predictions,top=1)

print(prediction_result[0][0])

print("completed execution")