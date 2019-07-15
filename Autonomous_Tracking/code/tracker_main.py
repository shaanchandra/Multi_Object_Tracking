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



def DETECT(model, layer_names, ct, frame, W, H, rgb):
    # instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject

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


def TRACK(ct, frame, rects, trackers, rgb):

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



def Main_Processor(frame, model, layer_names, rgb, ct, W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end, checkpoint_path):
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
        startX, startY, endX, endY, ct, trackers, start, end = DETECT(model, layer_names, ct, frame, W, H, rgb)


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

    return W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end



def Main_Handler(args, model, layer_names, path, checkpoint_path):
    # initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
    W = None
    H = None
    writer = None

    # initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down
    totalFrames = 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    ct = CentroidTracker(args.max_disappeared, args.max_distance)
    trackableObjects = {}
    df = []
    trackers = None
    startX, startY, endX, endY = 0,0,0,0
    start, end = 0,0

    if args.input_type == 1:

        vs = cv2.VideoCapture(path)
        if vs.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
        except:
            print("[!] WARNING ! Could not determine the No. of frames in the video. Can not estimate completion time")

        while True:
            # grab the next frame
            frame = vs.read()
            frame = frame[1]
    
            # if we are viewing a video and we did not grab a frame then we have reached the end of the video
            if frame is None:
                break
    
            # resize the frame to have a maximum width of 500 pixels, then convert the frame from BGR to RGB for dlib
            if args.resize:
                frame = imutils.resize(frame, args.im_width)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end = Main_Processor(frame, model, layer_names, rgb, ct, W, H, writer, totalFrames, trackers,
                                                                                                                   trackableObjects, df, startX, startY, endX, endY, start, end, checkpoint_path)

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
            W, H, writer, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end = Main_Processor(frame, model, layer_names, rgb, ct, W, H, writer, totalFrames, trackers,
                                                                                                                   trackableObjects, df, startX, startY, endX, endY, start, end, checkpoint_path)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and then update the FPS counter
            totalFrames += 1
            fps.update()
        
        # Stop the timer and display FPS information
        fps.stop()


    elap_estimate = end-start
    print("\n" + "-"*50 + "\n\t\tSTATISTICS:\n" + "-"*50)
    print("Total frames in video being processed =    ", total)
    print("Detection on single frame takes:      {:.4f} secs".format(elap_estimate))
    print("Estimated total time to finish (detection at every frame, no tracking):  {:.2f} secs".format(elap_estimate*total))
    print("Actual Elapsed time with tracking:       {:.2f} secs".format(fps.elapsed()))
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument("--prototxt", type = str, help="path to Caffe 'deploy' prototxt file", default = './model_checkpoint/MobileNetSSD_deploy.prototxt')
    parser.add_argument("--model", type = int, help = "Choose 0 for MobileNet and 1 for YOLOv3", default = 1)
    parser.add_argument("--model_path", type = str, help="path to Caffe pre-trained model", default = './model_checkpoint')
    parser.add_argument("--input_type", type=int, help="Choose 0 for image sequences and 1 for video", default = 1)
    parser.add_argument("--data_path", type=str, help="path to input video file", default = '../data/brain2.mp4')
    parser.add_argument("--output_path", type=str, help="path to optional output video file", default = './output_checkpoints/')

    # Tuning Parameters
    parser.add_argument("--max_disappeared", type=int, help = "Maximum frames a tracked object can be marked as 'disappeared' before forgetting it", default = 100)
    parser.add_argument("--resize", type = bool, help = "Whether to resize the frames for faster processing", default = True)
    parser.add_argument("--im_width", type = int, help = "Image Width for resizing the image", default = 350)
    parser.add_argument("--max_distance", type=int, help ="Maximum distance the object should have drifted from its previous position to mark it is disappeared and treat it as a new object", default = 100)
    parser.add_argument("--confidence", type=float, help="minimum probability to filter weak detections", default = 0.65)
    parser.add_argument("--thresh", type =float, help = "threshold when applying non-maximum suppression", default = 0.55)
    parser.add_argument("--skip_frames", type=int, help="# of skip frames between detections", default= 2)
    args = parser.parse_known_args()[0]

    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]



    # Assert the requried paths and correct combination of arguments
    if args.model == 0:
        if not os.path.exists(args.prototxt):
            sys.exit("[!] WARNING !! Prototext path does NOT exist!!")
    if not os.path.exists(args.model_path):
        sys.exit("[!] WARNING !! Model path does NOT exist!!")
    if not os.path.exists(args.data_path):
        sys.exit("[!] WARNING !! Data(input) path does NOT exist!!")
    if (args.data_path.endswith("mp4") or args.data_path.endswith("avi")) and args.input_type == 0:
        sys.exit("[!] WARNING !! Video file provided but processing mode is IMAGES")
    if args.data_path.endswith("/")  and args.input_type == 1:
        sys.exit("[!] WARNING !! Image file provided but processing mode is VIDEO")

    # Get the right model files based on the model selection
    if args.model == 0:
        model_path = os.path.join(args.model_path, 'MobileNetSSD_deploy.caffemodel')
    elif args.model == 1:
        wt_path  = os.path.join(args.model_path, 'yolov3.weights')
        config_path = os.path.join(args.model_path, 'yolov3.cfg')
    else:
        sys.exit("[!] WARNING !! Incorrect Model selection: Choose 0 for MobileNet and 1 for YOLOv3")



    # load our serialized model from disk
    print("="*80 + "\n\t\t\t\t TRACKING\n" + "="*80)
    print("\nLoading model...")

    if args.model == 0:
        model = cv2.dnn.readNetFromCaffe(args.prototxt, args.model_path)
        layer_names = None
    else:
        model = cv2.dnn.readNetFromDarknet(config_path, wt_path)
        layer_names = model.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    path = args.data_path
    checkpoint_path = args.output_path

    if not os.path.exists(checkpoint_path):
        print("Creating checkpoint path for output videos:  ", checkpoint_path)
        os.makedirs(checkpoint_path)
    else:
        print("Output videos written at: ", checkpoint_path)

    if not (args.input_type==0 or args.input_type == 1 ):
        sys.exit("[!] Incorrect Input Type argument: Choose 0 for image sequences and 1 for video")

    Main_Handler(args, model, layer_names, path, checkpoint_path)

    #################
    # Best Settings #
    #################

    # brain2.mp4 : 100, 75, 0.65, 0.55, 15


