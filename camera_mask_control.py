#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install python-vlc


# In[ ]:


import cv2
import sys
import tensorflow as tf
import vlc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
import math


#!/usr/bin/python
from PIL import Image
import os, sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



IMG_HEIGHT = 224
IMG_WIDTH = 224
#load model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.load_weights('my_model.h5')


face_cascade = cv2.CascadeClassifier('face.xml')
sound_path = './ambulance.mp3'
sound = vlc.MediaPlayer(sound_path)
sound_playing = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 22) #1280, 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 544) #720, 480
    
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pred=0
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 5) #detect faces
    no_masks = []
    for (x,y,w,h) in faces:
        sub_img = gray[y:y+h, x:x+w] #take part of image that contains the face only
        cv2.imwrite('deleted.jpg', sub_img) #save part of image that contains the face only
        im = Image.open('deleted.jpg')
        imResize = im.resize((224,224), Image.ANTIALIAS)
        imResize.save('deleted' + '.jpg', 'JPEG', quality=90)
        test_image = image.load_img('deleted.jpg'
                   , target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)


    
        pred = sigmoid(result[0])
        #pred = model.predict(img) # predict
       
        mask_label = 'No Mask! :(' if pred==1 else 'Mask ON :)'
        color = (0, 0, 255) if pred==1 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3) #draw box around detected face
        cv2.putText(img=frame, text=mask_label, org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color)
    #check if at least one person is not wearing a mask. If so, play alert sound!
    if pred==1:
        if sound_playing == False:
            sound_playing = True
            sound.play()
    else:
        if sound_playing == True:
            sound_playing = False
            sound.stop()
        
    cv2.imshow('Driver_frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




