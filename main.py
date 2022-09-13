from email import iterators
import cv2
import pickle
import numpy as np
import cvzone

cap =cv2.VideoCapture('./data/jackson_trim.mp4')
width, height= 120, 80

with open('./data/position_list2','rb') as f:
    posList = pickle.load(f) 

def check_parking_space(img_process):

    space_counter=0
    for pos in posList:
        x,y = pos
        # cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),2)

        img_crop = img_process[y:y+height,x:x+width]
        # cv2.imshow(str(x*y),img_crop)
        count =cv2.countNonZero(img_crop)
        cvzone.putTextRect(img,str(count),(x,y+height-2),scale=1,thickness=2,offset=0,colorR=(0,0,255))

        if count<1000:
            color = (0,255,0)
            thickness = 4
            space_counter+=1
        else:
            color = (0,0,255)
            thickness=2
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),color,thickness)

    cvzone.putTextRect(img,"Free Space: "+str(space_counter)+" / "+str(len(posList)),(100,50),scale=3,thickness=3,offset=20)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)

while cap.isOpened():
    success,img = cap.read()

    if cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    img_gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur =cv2.GaussianBlur(img_gray,(3,3),1)

    img_threshold = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    img_median= cv2.medianBlur(img_threshold,5)

    kernal = np.ones((3,3),np.uint8)
    img_dilate = cv2.dilate(img_median,kernal,iterations=1)

    check_parking_space(img_dilate)

    result.write(img)
        
    cv2.imshow('image',img)
    cv2.imshow('imageBlur',img_blur)

    

    cv2.imshow('imageThreshold',img_threshold)
    # cv2.imshow('imageMedian',img_median)
    cv2.imshow('imageDilate',img_dilate)


    cv2.waitKey(1)
