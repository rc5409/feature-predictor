import os
import numpy as np
import tensorflow
import PIL
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input , decode_predictions
from tensorflow.keras.models import load_model



model=tensorflow.keras.models.load_model('mymodel.h5')
print("model is loaded\n")


def getpredictions(filename):
    imgpath='Uploaded_temp/'+filename
    img=load_img(imgpath,target_size=(224,224))
    img=img_to_array(img)
    img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    img=preprocess_input(img)

    predictions=model.predict(img)
    prediction_result=decode_predictions(predictions,top=1)
    return prediction_result[0][0]

