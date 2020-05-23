import numpy as np
import os
import tensorflow as tf
import cv2
import dlib
import math

from utils import visualization_utils as vis_util

carCascade = cv2.CascadeClassifier('myhaar.xml')

WIDTH = 1280
HEIGHT = 720

cap = cv2.VideoCapture('cars.mp4')
#cap = cv2.VideoCapture('1.mp4')
#cap = cv2.VideoCapture('C:\\Users\\Pc\\Downloads\\tr2.mp4')


#MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb') 
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt') 

category_index = {      1: {'id': 1, 'name': 'person'},
                        2: {'id': 2, 'name': 'bicycle'},
                        3: {'id': 3, 'name': 'car'},
                        4: {'id': 4, 'name': 'motorcycle'},
                        5: {'id': 5, 'name': 'airplane'},
                        6: {'id': 6, 'name': 'bus'},
                        7: {'id': 7, 'name': 'train'},
                        8: {'id': 8, 'name': 'truck'}}


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def speedFinder(firstLoc, lastLoc):
	pixelsDistance = math.sqrt(math.pow(lastLoc[0] - firstLoc[0], 2) + math.pow(lastLoc[1] - firstLoc[1], 2))
	pixelsPerMetre = 8.8
	metersDistance = pixelsDistance / pixelsPerMetre
	fps = 10
	speed = metersDistance * fps * 3.6
	return speed	

# Detection
def object_detection_function(): 

    frameCounter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    currentCarID = 0
    carTracker = {}
    initialCarLocation = {}
    finalCarLocation = {}
    speed = [None] * 1000

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            while cap.isOpened():
                (ret, frame) = cap.read()
                if not ret:
                    print ('end of the video file...')
                    break
                
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                frameCounter = frameCounter + 1	
                carIDtoDelete = []
                cv2.putText(frame, "{0} vehicles passed".format(currentCarID), (10, 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                

                for carID in carTracker.keys():
                    trackingQuality = carTracker[carID].update(frame)                   
                    if trackingQuality < 3:
                        carIDtoDelete.append(carID)
                
                    
                if (frameCounter % 10):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cars = carCascade.detectMultiScale(gray, 1.4, 13, 18, (24, 24))
                    
                    for (_x, _y, _w, _h) in cars:
                        x = int(_x)
                        y = int(_y)
                        w = int(_w)
                        h = int(_h)                        
                        x_bar = x + 0.5 * w
                        y_bar = y + 0.5 * h                        
                        matchCarID = None
                                                                      
                        for carID in carTracker.keys():
                            trackedPosition = carTracker[carID].get_position()
                            
                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())                           
                            t_x_bar = t_x + 0.5 * t_w
                            t_y_bar = t_y + 0.5 * t_h
                            
                            if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                matchCarID = carID
                                
                        if matchCarID is None:
                            print ('Creating new tracker ' + str(currentCarID))
                            
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))                            
                            carTracker[currentCarID] = tracker
                            initialCarLocation[currentCarID] = [x, y, w, h]                           
                            currentCarID = currentCarID + 1
                            
                
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    
                    start_x = int(trackedPosition.left())
                    start_y = int(trackedPosition.top())
                    end_w = int(trackedPosition.width())
                    end_h = int(trackedPosition.height())                   
                    #speed estimation
                    finalCarLocation[carID] = [start_x, start_y, end_w, end_h]
                
                for i in initialCarLocation.keys():
                    if frameCounter % 1 == 0:   
                        [x1, y1, w1, h1] = initialCarLocation[i]
                        [x2, y2, w2, h2] = finalCarLocation[i]                        
                        # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                        initialCarLocation[i] = [x2, y2, w2, h2]                        
                        # print 'new previous location: ' + str(carLocation1[i])
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            if (speed[i] == None or speed[i] == 0):
                                speed[i] = speedFinder([x1, y1, w1, h1], [x2, y2, w2, h2])                              

                            if speed[i] != None and y1 >= 210 and x1 >= 100 and w1 >= 120:
                                cv2.putText(frame, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1+80)), font, 0.5, (255, 255, 255), 2)
                                cv2.putText(frame, "ID="+str(carID), (int(x1 + w1/2), int(y1+100)), font, 0.5, (255, 0, 255), 2)            
                        
                  
                image_np_expanded = np.expand_dims(frame, axis=0)
                (boxes, scores, classes, num) = \
                                    sess.run([detection_boxes, detection_scores,
                                              detection_classes, num_detections],
                                              feed_dict={image_tensor: image_np_expanded})
              
                (boxes, labels) = \
                                    vis_util.visualize_boxes_and_labels_on_image_array(
                                            cap.get(1),
                                            frame,
                                            np.squeeze(boxes),
                                            np.squeeze(classes).astype(np.int32),
                                            np.squeeze(scores),
                                            category_index,
                                            use_normalized_coordinates=True,
                                            line_thickness=4,
                                            )
                    
                                #detection
#                                image_np_expanded = np.expand_dims(frame, axis=0)
#                                (boxes, scores, classes, num) = \
#                                    sess.run([detection_boxes, detection_scores,
#                                              detection_classes, num_detections],
#                                              feed_dict={image_tensor: image_np_expanded})
#                                #boxes
#                                (counter, csv_line) = \
#                                    vis_util.visualize_boxes_and_labels_on_image_array(
#                                            cap.get(1),
#                                            frame,
#                                            np.squeeze(boxes),
#                                            np.squeeze(classes).astype(np.int32),
#                                            np.squeeze(scores),
#                                            category_index,
#                                            use_normalized_coordinates=True,
#                                            line_thickness=4,
#                                            )

                cv2.imshow('Traffic Surveillance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):                    
                    break

            cap.release()
            cv2.destroyAllWindows()
object_detection_function()		