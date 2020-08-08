#!/usr/bin/env python
# coding: utf-8

import pickle
model = pickle.load(open('finalized_model.sav', 'rb'))

import cv2
import numpy as np

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture('path') # 0 for live camera and path for precorded video
labels_dict=['MASK','NO MASK']
color_dict={0:(0,255,0),1:(0,0,255)}
ret,img=source.read()
while(True):

    ret,img=source.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.3,5)  #converted gray to img

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(128,128))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,128,128,3))
        result=model.predict(reshaped)

        #label=np.argmax(result,axis=1)[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[np.argmax(result)],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[np.argmax(result)],-1)
        cv2.putText(img, labels_dict[np.argmax(result)], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
source.release()       
cv2.destroyAllWindows()
