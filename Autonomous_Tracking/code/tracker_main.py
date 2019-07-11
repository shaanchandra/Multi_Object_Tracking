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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")

import os



def DETECT(model, ct, frame, W, H, rgb):
    # instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    # ct = CentroidTracker(maxDisappeared=50, maxDistance=100)
    trackers = []
    startX, startY, endX, endY = 0,0,0,0


    # convert the frame to a blob and pass the blob through the network and obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    model.setInput(blob)
    detections = model.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum confidence
        if confidence > args.confidence:
            # extract the index of the class label from the detections list
            idx = int(detections[0, 0, i, 1])

#             # if the class label is not a person, ignore it
#             if CLASSES[idx] != "person":
#                 continue

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

            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
    return startX, startY, endX, endY, ct, trackers


def TRACK(ct, frame, rects, trackers, rgb):

    # loop over the trackers
    for tracker in trackers:

        # update the tracker and grab the updated position
        tracker.update(rgb)
        # (success, box) = tracker.update(rgb)
        pos = tracker.get_position()

        # # unpack the position object
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



def Main_Handler(args, model, path, checkpoint_path):
    # initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
    W = None
    H = None
    writer = None

    # initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down
    totalFrames = 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    ct = CentroidTracker(maxDisappeared=50, maxDistance=100)
    trackableObjects = {}
    df = []
    trackers = None
    startX, startY, endX, endY = 0,0,0,0

    if args.input_type == 1:

        vs = cv2.VideoCapture(path)
        if vs.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

        while True:
            # grab the next frame
            frame = vs.read()
            frame = frame[1]
    
            # if we are viewing a video and we did not grab a frame then we have reached the end of the video
            if frame is None:
                break
    
            # resize the frame to have a maximum width of 500 pixels, then convert the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=700)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY = Main_Processor(frame, rgb, ct, W, H, writer, totalFrames, trackers,
                                                                                                                   trackableObjects, df, startX, startY, endX, endY, checkpoint_path)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and then update the FPS counter
            totalFrames += 1
            fps.update()

            # Stop the timer and display FPS information
            fps.stop()
        vs.release()

    else:
        path = os.path.join(path, '*.jpg')
        image_files = sorted(glob.glob(path))
        # loop over frames from the video stream
        for image in image_files:
            frame = cv2.imread(image)
    
            # resize the frame to have a maximum width of 500 pixels, then convert the frame from BGR to RGB for dlib
            # frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY = Main_Processor(frame, rgb, ct, W, H, writer, totalFrames, trackers,
                                                                                                                   trackableObjects, df, startX, startY, endX, endY, checkpoint_path)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and then update the FPS counter
            totalFrames += 1
            fps.update()
        
            # Stop the timer and display FPS information
            fps.stop()

    print("\nSTATISTICS:\n" + "-"*30)
    print("Elapsed time:  {:.2f}".format(fps.elapsed()))
    print("Approx. FPS:   {:.2f}".format(fps.fps()))
    df = pd.DataFrame(np.matrix(df), columns = ['frame', 'ID','start_x','start_y', 'end_x', 'end_y', 'centroid_x', 'centroid_y'])
    df.to_csv('tracked.csv')
    df = pd.read_csv('tracked.csv')
    df.describe()
    df.head()
    print("Number of unique objects detected and tracked:  ", len(df['ID'].unique()))

    for idx, ID in enumerate(np.unique(df['ID'])):
        df['ID'][df['ID'] == ID] = idx

    plt.figure(figsize=(8,8))
    plt.scatter(df['centroid_x'], df['centroid_y'], c=df['ID'], cmap='jet')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.tight_layout()
    plt.savefig('tracked_vis.png', format='png', dpi=300)
    plt.show()

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # close any open windows
    cv2.destroyAllWindows()



def Main_Processor(frame, rgb, ct, W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, checkpoint_path):
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

      # Initialize the writer to construct the output video
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(checkpoint_path + 'out.mp4', fourcc, 30, (W, H), True)

    # initialize the current status along with our list of bounding box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive object detection method to aid our tracker
    if totalFrames % args.skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        startX, startY, endX, endY, ct, trackers = DETECT(model, ct, frame, W, H, rgb)


    # otherwise, we should utilize our object *trackers* rather than object *detectors* to obtain a higher frame processing throughput
    else:
        status = "Tracking"
        rects = TRACK(ct, frame, rects, trackers,rgb)

    # use the centroid tracker to associate the (1) old object centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # Write text and mark centroids on the frames
    for (objectID, centroid) in objects.items():
        track_obj = trackableObjects.get(objectID, None)

        if track_obj is None:
            track_obj = TrackableObject(objectID, centroid)

        trackableObjects[objectID] = track_obj


        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        cv2.putText(frame, str(totalFrames), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, status, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        df.append([totalFrames, objectID, startX, startY, endX, endY, centroid[0], centroid[1]])


    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)

    return W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototxt", type = str, help="path to Caffe 'deploy' prototxt file", default = './model_checkpoint/MobileNetSSD_deploy.prototxt')
    parser.add_argument("--model", type = str, help="path to Caffe pre-trained model", default = './model_checkpoint/MobileNetSSD_deploy.caffemodel')
    parser.add_argument("--data_path", type=str, help="path to optional input video file", default = '../data/crowd1/')
    parser.add_argument("--output_path", type=str, help="path to optional output video file", default = './output_checkpoints/')
    parser.add_argument("--input_type", type=int, help="Choose 0 for image sequences and 1 for video", default = 0)
    parser.add_argument("--confidence", type=float, help="minimum probability to filter weak detections", default = 0.7)
    parser.add_argument("--skip_frames", type=int, help="# of skip frames between detections", default=20)
    args = parser.parse_known_args()[0]

    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    # Assert the requried paths and correct combination of arguments
    if not os.path.exists(args.prototxt):
        sys.exit("[!] WARNING !! Prototext path does NOT exist!!")
    if not os.path.exists(args.model):
        sys.exit("[!] WARNING !! Model path does NOT exist!!")
    if not os.path.exists(args.data_path):
        sys.exit("[!] WARNING !! Data(input) path does NOT exist!!")
    if (args.data_path.endswith("mp4") or args.data_path.endswith("avi")) and args.input_type == 0:
        sys.exit("[!] WARNING !! Video file provided but processing mode is IMAGES")
    if args.data_path.endswith("/")  and args.input_type == 1:
        sys.exit("[!] WARNING !! Image file provided but processing mode is VIDEO")



    # load our serialized model from disk
    print("="*80 + "\n\t\t\t\t TRACKING\n" + "="*80)
    print("\nLoading model...")
    model = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    path = args.data_path
    checkpoint_path = args.output_path

    if not os.path.exists(checkpoint_path):
        print("Creating checkpoint path for output videos:  ", checkpoint_path)
        os.makedirs(checkpoint_path)
    else:
        print("Output videos written at: ", checkpoint_path)

    if not (args.input_type==0 or args.input_type == 1 ):
        sys.exit("[!] Incorrect Input Type argument: Choose 0 for image sequences and 1 for video")

    Main_Handler(args, model, path, checkpoint_path)


