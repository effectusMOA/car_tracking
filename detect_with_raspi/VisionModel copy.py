import numpy as np
import cv2
import os
import sys
import glob
import random
import importlib.util
import math
from tensorflow.lite.python.interpreter import Interpreter
from Msg import Msg

import matplotlib
import matplotlib.pyplot as plt

import Processlib

class VisionModel(Processlib.Processlib):
    def __init__(self, main_queue, message_queue):
        super().__init__("VisionModel", main_queue, message_queue)
        self._cap = None
    
    def stop(self):
        super()
        
        self._cap.release()
        cv2.destroyAllWindows()

    def run(self):
        super()

        modelpath='detect_edgetpu.tflite'
        lblpath='labelmap.txt'
        min_conf=0.5
        # cap = cv2.VideoCapture('demo.mp4')
        self._cap = cv2.VideoCapture('license.mp4')

        interpreter = Interpreter(model_path=modelpath)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        float_input = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        with open(lblpath, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        count = 0
        center_points_prev_frame = []

        tracking_objects = {}
        track_id = 0
        max_age=10
        # detections = []
        stranger_id={}

        RFID_loc=(1,1)
        Becon_loc=(500,500)
        stranger_pt={}
        
        count =0
        
        find_person_id={}
        print_person={}
        person_index=0

        
        find_parcel_id={}
        print_parcel={}
        parcel_index=0

        while self._running:
            count+=1
            # message = Msg()
            # if (not self._message_queue.empty()):
            #     message: Msg = self._message_queue.get()
                
            # if (message.msg_from == "RfidReader"):
            #     if (message.msg == "in"):
            #         pass
            #     elif (message.msg == "out"):
            #         pass

            # elif (message.msg_from == "BeaconDetector"):
            #     if (message.msg == "in"):
            #         pass
            #     elif (message.msg == "out"):
            #         pass

            # elif (message.msg_from == "GyroAnalysis"):
            #     message.msg_from = "VisionModel"
            #     message.msg_to = "NotifyModel"
            #     message.photo = "<< you need to worker photo here!!"
            #     self._main_queue(message)

            ret, frame = self._cap.read()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if float_input:
                input_data = (np.float32(input_data) - input_mean) / input_std
                
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

            parcel_detections=[]
            person_detections=[]
            
            #frame check

            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0) and (labels[int(classes[i])]=='LicensePlate')): #후에 Parcel로 변경
                    print('person detect')
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    cx= int((xmin+xmax)/2)
                    cy= int((ymin+ymax)/2)
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    # cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 255), 2)
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    person_detections.append((cx,cy)) # 0 : max_age값 False: find_parcel_id에서 아직 감지가 되지 않았다.
                
                if ((scores[i] > min_conf) and (scores[i] <= 1.0) and (labels[int(classes[i])]=='person')):
                    
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    cx= int((xmin+xmax)/2)
                    cy= int((ymin+ymax)/2)
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    # cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 255), 2)
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    parcel_detections.append((cx,cy)) # 0 : max_age값 False: find_parcel_id에서 아직 감지가 되지 않았다.
                
            #object detect / person_detections->find_person_id  
            
            parcel_detections_copy=parcel_detections.copy()
            find_parcel_id_copy=find_parcel_id.copy()
            person_detections_copy=person_detections.copy()
            find_person_id_copy=find_person_id.copy()


            for parcel_id, info in find_parcel_id_copy.items():
                
                object_exists= False
                for pt in parcel_detections_copy:
                    
                    distance = math.hypot(info[0]-pt[0],info[1]-pt[1])
                    
                    if distance < 150:
                        
                        find_parcel_id[parcel_id]=(pt[0],pt[1],0,info[3])
                        object_exists = True
                        if pt in parcel_detections:
                            parcel_detections.remove(pt)
                        continue
                if not object_exists:
                    print('before dont exitst')
                    age=info[2]+1
                    if age > max_age:
                        find_parcel_id.pop(parcel_id)
                    else:
                        find_parcel_id[parcel_id]=(info[0],info[1],age,info[3])

            for pt in parcel_detections:
                
                find_parcel_id[parcel_index] = (pt[0],pt[1],0,'safe')
                parcel_index += 1


            print(person_detections)

            for person_id, info in find_person_id_copy.items():
                object_exists= False
                for pt in person_detections_copy:
                        
                    distance = math.hypot(info[0]-pt[0],info[1]-pt[1])
                    print(distance)
                    
                    if distance < 150:
                        find_person_id[person_id]=(pt[0],pt[1],0,info[3])
                        object_exists = True
                        print('pt',pt)
                        print('person_det',person_detections)
                        if pt in person_detections:
                            person_detections.remove(pt)
                            print('remove',pt)
                if not object_exists:
                    
                    age=info[2]+1
                    if age > max_age:
                        find_person_id.pop(person_id)
                    else:
                        find_person_id[person_id]=(info[0],info[1],age,info[3])

            print(person_detections)
            
            for pt in person_detections:
                print('new id')
                find_person_id[person_index] = (pt[0],pt[1],0,'stranger')
                person_index += 1

            
            #set interrupt device

            
            for object_id, pt in find_person_id.items():
                if pt[3]=='stranger':
                    if object_id in stranger_pt:
                        stranger_pt[object_id]+=1
                        if stranger_pt[object_id] > 3000 and object_id not in stranger_id:
                            stranger_id.append(stranger_id)
                            cv2.imwrite('stranger.jpg',frame)
                            message.msg_from = "VisionModel"
                            message.msg_to = "NotifyModel"
                            message.photo = "stranger.jpg"
                            self._main_queue.put(message)
            
            uuid=1000
            
            if (count == 10):
                
                min_RFID_distance=10000
                tag_id=None
                for object_id, pt in find_person_id.items():
                    distance = math.hypot(RFID_loc[0]-pt[0],RFID_loc[1]-pt[1])
                    if distance < min_RFID_distance:
                        min_RFID_distance = distance
                        tag_id=object_id
                        tag_info=pt
                find_person_id[uuid] = (tag_info[0],tag_info[1],tag_info[2],'worker') 
                del find_person_id[tag_id]
                # print(find_person_id)
            print('RFid taging',find_person_id)        
                    
                    

            while (self._running and not self._message_queue.empty()):
                message: Msg = self._message_queue.get()
                
                if (message.msg_from == "RfidReader"):
                    print(Msg.uuid)
                    print(find_person_id)
                    min_RFID_distance=10000
                    tag_id=None
                    for object_id, pt in find_person_id.items():
                        distance = math.hypot(RFID_loc[0]-pt[0],RFID_loc[1]-pt[1])
                        if distance < min_RFID_distance:
                            min_RFID_distance = distance
                            tag_id=object_id
                            tag_info=pt
                    find_person_id[int(Msg.uuid)] = (pt[0],pt[1],pt[2],'worker') 
                    del find_person_id[tag_id]
                    print(find_person_id)
                    
                    if (message.msg == "in"):
                        pass
                    elif (message.msg == "out"):
                        pass
                
                elif (message.msg_from == "BeaconDetector"):
                    print(Msg.uuid)
                    print(find_parcel_id)
                    min_Becon_distance=10000
                    tag_id=None
                    for object_id, pt in find_parcel_id.items():
                        distance = math.hypot(Becon_loc[0]-pt[0],Becon_loc[1]-pt[1])
                        if distance < min_Becon_distance:
                            min_Becon_distance = distance
                            tag_id=object_id
                            tag_info=pt
                    find_parcel_id[int(Msg.uuid)] = pt 
                    del find_parcel_id[tag_id]
                    print(find_parcel_id)
                    # if (message.msg == "in"):
                    #     pass
                    # elif (message.msg == "out"):
                    #     pass
                

                elif (message.msg_from == "GyroAnalysis"):
                    cv2.imwrite('accident.jpg',frame)
                    message.msg_from = "VisionModel"
                    message.msg_to = "NotifyModel"
                    message.photo = "accident.jpg"
                    self._main_queue.put(message)

            print('after Message', find_person_id)
            
            # find_id 출력

            for object_id, pt in find_parcel_id.items():
                if (pt[2]==0):
                    if (pt[3]=='safe'):
                        cv2.circle(frame, (pt[0],pt[1]), 3, (0, 255, 255), -1)
                        cv2.putText(frame, (str(object_id)+pt[3]), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)
  
                    elif (pt[3]=='shock'):
                        cv2.circle(frame, (pt[0],pt[1]), 3, (0, 255, 255), -1)
                        cv2.putText(frame, (str(object_id)+pt[3]), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)
                        
                    
            for object_id, pt in find_person_id.items():
                if (pt[2]==0):
                    if (pt[3]=='stranger'):
                        cv2.circle(frame, (pt[0],pt[1]), 3, (0, 255, 255), -1)
                        cv2.putText(frame, (str(object_id)+pt[3]), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)    
                    if (pt[3]=='worker'):
                        cv2.circle(frame, (pt[0],pt[1]), 3, (0, 255, 255), -1)
                        cv2.putText(frame, (str(object_id)+pt[3]), (pt[0], pt[1] - 7), 0, 1, (0, 255, 255), 2)    

            # print(detections)
            #         label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            #         labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            #         label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            #         cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            #         cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            cv2.imshow('output',frame)
            if(cv2.waitKey(0) & 0xFF == ord('q')):
                break



