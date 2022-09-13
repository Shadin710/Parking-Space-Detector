from pydoc import classname
from sre_constants import SUCCESS
import cv2
import numpy as np


cap = cv2.VideoCapture('./data/jackson.mp4')
classFiles = './yolo/coco.names'
class_names=[]
whT=320
confidence_threshold = 0.3
nms_threshold = 0.3

def find_object(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence>confidence_threshold and classId==2:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confidence_threshold,nms_threshold)
    # print(indices)
    # for i in indices:
    #     box = bbox[i]
    #     x,y,w,h = box[0],box[1],box[2],box[3]

    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
    #     cv2.putText(img,f'{class_names[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
    
    return indices, bbox,classIds,confs


def similar_score(arr1,arr2):

    try:
        score = 1
        for a1,a2 in zip(arr1,arr2):
            score = score*abs(a1-a2)
        return score
    except:
        pass
        
def intersect(abs_parking,parking):

    # [[],[],[],[]] abs
    # [[],[],[],[]] parking

    if len(abs_parking)==0:
        return parking

    abs_parking_new = []
    for element in parking:
        sim_score = []
        for abs_element in abs_parking:
            sim_score.append(similar_score(element,abs_element))
        try:
            min_score = min(sim_score)
            for i in range(len(sim_score)):
                if min_score ==sim_score[i]:
                    get_index = i
                    break
            abs_parking_new.append(abs_element[get_index])
        except:
            pass
    return abs_parking_new


with open(classFiles,'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')


model_cfg = './yolo/yolov3-320.cfg'
model_weights = './yolo/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_cfg,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

car_parking_arr = []
abs_car_parking =[]
counter=0
new_indx =[]
while cap.isOpened():
    success, img = cap.read()
    

    if success:
        blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop = False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        outputs  = net.forward(output_names)

        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)

        indices,bbox,classIds,confs = find_object(outputs,img)


        if cap.get(cv2.CAP_PROP_POS_FRAMES)%20==0 and  cap.get(cv2.CAP_PROP_POS_FRAMES)<int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2):
            for i in indices:
                box = bbox[i]
                x,y,w,h = box[0],box[1],box[2],box[3]

                car_parking_arr.append([x,y,w,h])

            abs_car_parking = intersect(abs_car_parking[:],car_parking_arr[:])
            counter+=1

        print(abs_car_parking)
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES)>=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2):
            for i in abs_car_parking:
                box = abs_car_parking[i]
                try:
                    x,y,w,h = box[0],box[1],box[2],box[3]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                    cv2.putText(img,f'{class_names[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                except:
                    pass

        # if cap.get(cv2.CAP_PROP_POS_FRAMES)%20==0:
        #     for i in indices:
        #         box = bbox[i]
        #         x,y,w,h = box[0],box[1],box[2],box[3]

        #         car_parking_arr.append([x,y,w,h])

        #     abs_car_parking = intersect(abs_car_parking,car_parking_arr)

        cv2.imshow('Image',img)
        cv2.waitKey(1)
    else:
        break

cap.release()
