from centroid_tracker.centroidtracker import CentroidTracker
from centroid_tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import pandas as pd
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import glob

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")

import os

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]


def DETECT(args, model, layer_names, ct, frame, W, H, rgb):
    # instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    # ct = CentroidTracker(maxDisappeared=50, maxDistance=100)
    trackers = []
    # trackers = cv2.MultiTracker_create()
    startX, startY, endX, endY = 0,0,0,0


    # convert the frame to a blob and pass the blob through the network and obtain the detections
    if args.model == 0:
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        model.setInput(blob)
        start = time.time()
        detections = model.forward()
        end = time.time()
    else:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
        model.setInput(blob)
        start = time.time()
        layer_outs = model.forward(layer_names)
        end = time.time()

    # loop over the detections
    if args.model == 0:
        for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by requiring a minimum confidence
            if confidence > args.confidence:
                # extract the index of the class label from the detections list
                idx = int(detections[0, 0, i, 1])
    
                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue
    
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
    
                # construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
                correl_tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                correl_tracker.start_track(rgb, rect)
                # correl_tracker = cv2.TrackerKCF_create()
    
                # add the tracker to our list of trackers so we can utilize it during skip frames
                trackers.append(correl_tracker)
                # initBB = ([startX, startY, startX-endX, startY-endY])
                # trackers.add(correl_tracker, frame, initBB)
    
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(frame, str(confidence), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        boxes = []
        confidences = []
        # loop over each of the layer outputs
        for out in layer_outs:
            # loop over each detections
            for detection in out:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                confidence = scores[np.argmax(scores)]

                # filter out weak detections by requiring a minimum confidence
                if confidence > args.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    startX = x
                    startY = y
                    endX = x+width
                    endY = y+height

                    # construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
                    correl_tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    correl_tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can utilize it during skip frames
                    trackers.append(correl_tracker)

                    boxes.append([x,y, int(width), int(height)])
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence,args.thresh)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
    
                # draw a bounding box rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(confidences[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    return startX, startY, endX, endY, ct, trackers, start, end



def TRACK(args, ct, frame, rects, trackers, rgb):

    # loop over the trackers
    for tracker in trackers:

        # update the tracker and grab the updated position
        tracker.update(rgb)
        # (success, box) = tracker.update(rgb)
        pos = tracker.get_position()

        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        # (x, y, w, h) = [int(v) for v in box]
        # cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

        # add the bounding box coordinates to the rectangles list
        rects.append((startX, startY, endX, endY))
        # rects.append((x,y,w,h))
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
    return rects
    # (success, box) = trackers.update(rgb)
    # for b in box:
    #     (x, y, w, h) = [int(v) for v in b]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
    #     rects.append((x,y,w,h))
    # return rects