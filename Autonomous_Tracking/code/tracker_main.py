from centroid_tracker.centroidtracker import CentroidTracker
from centroid_tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from utils import DETECT, TRACK

import pandas as pd
import numpy as np
import argparse
import configargparse
import imutils
import time
import dlib
import cv2
import glob
import yaml

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")

import os


### Used by Main_Handler to apply DETECT or TRACK and generate tracked frames
def Main_Processor(frame, model, layer_names, rgb, orig_frame, thresh, ct, W, H, writer_orig, writer_thresh, writer_bilat, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end, checkpoint_path):
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

      # Initialize the writer to construct the output video
    if writer_orig is None and writer_thresh is None and writer_bilat is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer_orig = cv2.VideoWriter(checkpoint_path + 'out_orig.mp4', fourcc, args.fps, (W, H), True)
        writer_thresh = cv2.VideoWriter(checkpoint_path + 'out_thresh.mp4', fourcc, args.fps, (W, H), 0)
        writer_bilat = cv2.VideoWriter(checkpoint_path + 'out_bilat.mp4', fourcc, args.fps, (W, H), True)

    # initialize the current status along with our list of bounding box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive object detection method to aid our tracker
    if totalFrames % args.skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        startX, startY, endX, endY, ct, trackers, start, end = DETECT(args, model, layer_names, ct, frame, orig_frame, thresh, W, H, rgb)

    # otherwise, we should utilize our object *trackers* rather than object *detectors* to obtain a higher frame processing throughput
    else:
        status = "Tracking"
        rects = TRACK(args, ct, frame, orig_frame, thresh, rects, trackers, rgb)

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
        cv2.putText(frame, str(totalFrames), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, status, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(orig_frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(orig_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        cv2.putText(orig_frame, str(totalFrames), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(orig_frame, status, (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        df.append([totalFrames, objectID, startX, startY, endX, endY, centroid[0], centroid[1]])

    if writer_orig is not None and writer_thresh is not None and writer_bilat is not None:
        writer_orig.write(orig_frame)
        writer_thresh.write(thresh)
        writer_bilat.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Orig_Frame", orig_frame)
    time.sleep(0.05)

    return W, H, writer_orig, writer_thresh, writer_bilat, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end




### Based on video or image sequences as input, this block applies appropriate read, write and pre-processing steps
def Main_Handler(args, model, layer_names, path, checkpoint_path):
    # initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
    W = None
    H = None
    writer_orig, writer_thresh, writer_bilat = None, None, None

    # initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down
    totalFrames = 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    ct = CentroidTracker(args.max_disappeared, args.max_distance)
    trackableObjects = {}
    df = []
    trackers = None
    total = -1
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
            print("[!] WARNING ! Could not determine the No. of frames in the video. Can not estimate completion time\n\n")
        c = 0
        while True:
            # grab the next frame
            frame = vs.read()
            frame = frame[1]
            c+=1
            # if c<1500:
            #     continue
    
            # if we are viewing a video and we did not grab a frame then we have reached the end of the video
            if frame is None:
                break
    
            # resize the frame to have a maximum width of 500 pixels, then convert the frame from BGR to RGB for dlib
            if args.resize:
                frame = imutils.resize(frame, width=args.im_width)

            # Apply Gaussian/BiLateral blur to smooth out the background
            ### cv2.GaussianBlur(img, (kernel_size), sigmas)
            ### If one given, other also equal to this. If 0 given then calcul from the kernel size
            # frame = cv2.GaussianBlur(frame, (5,5), 0)

            ### cv2.bilateralFilter(src_img, diameter, sigmaColor, sigmaSpace)
            ### Higher SC means farther colors in the neigh will be mixed together resulting in larger areas of sem-equal color
            ### Higher SS means farther pixels will influence each other as long as colors close enough. If d given, that is
            ### taken as neigh size else d propr to SS
            cv2.imshow("Orig Frame", frame)
            frame = cv2.bilateralFilter(frame, args.diam, args.sigma_color, args.sigma_space)

            # Segmentation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Gray", gray)
            # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, args.block_size, args.constant)
            cv2.imshow("Thresh", thresh)

            # Further noise removal
            kernel = np.ones((3, 3), np.uint8)
            denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            # cv2.imshow("Denoised-Thresh", denoised)

            # # Finding sure foreground area
            # dist_transform = cv2.distanceTransform(denoised, cv2.DIST_L2, 5)
            # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            # cv2.imshow("Foreground", sure_fg)
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
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
        except:
            print("[!] WARNING ! Could not determine the No. of frames in the video. Can not estimate completion time")

        # loop over frames from the video stream
        count=0
        for image in image_files:
            orig_frame = cv2.imread(image)
            count+=1

            if orig_frame is None:
                break

            if count<1200:
                continue
            # resize the frame to have a maximum width of 500 pixels, then convert the frame from BGR to RGB for dlib
            if args.resize:
                orig_frame = imutils.resize(orig_frame, width = args.im_width)

            # Apply Gaussian/BiLateral blur to smooth out the background
            if args.filter_type == 0:
                ### cv2.GaussianBlur(img, (kernel_size), sigmas)
                ### sigmas: If one given, other also equal to this. If 0 given then calcul from the kernel size
                cv2.imshow("Orig Frame", orig_frame)
                frame = cv2.GaussianBlur(orig_frame, (11,11), 0)

            else:
                ### cv2.bilateralFilter(src_img, diameter, sigmaColor, sigmaSpace)
                ### Higher SC means farther colors in the neigh will be mixed together resulting in larger areas of sem-equal color
                ### Higher SS means farther pixels will influence each other as long as colors close enough. If d given, that is
                ### taken as neigh size else d propr to SS
                # cv2.imshow("Orig Frame", orig_frame)
                frame = cv2.bilateralFilter(orig_frame, args.diam, args.sigma_color, args.sigma_space)

            # Segmentation by Adaptive Thresholding
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Gray", gray)
            # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, args.block_size, args.constant)
            cv2.imshow("Thresh", thresh)

            # Further noise removal
            kernel = np.ones((3, 3), np.uint8)
            denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            # cv2.imshow("Denoised-Thresh", denoised)

            # # Finding sure foreground area
            # dist_transform = cv2.distanceTransform(denoised, cv2.DIST_L2, 5)
            # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            # cv2.imshow("Foreground", sure_fg)

            W, H, writer_orig, writer_thresh, writer_bilat, totalFrames, trackers, trackableObjects, df, startX, startY, endX, endY, start, end = Main_Processor(frame, model, layer_names, rgb, orig_frame, thresh, ct, W, H, writer_orig, writer_thresh, writer_bilat, totalFrames, trackers,
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
    print("\n" + "-"*80 + "\n\t\t\t\t STATISTICS \n" + "-"*80)
    print("\nTotal frames in video being processed:                                  ", total)
    print("Detection on single frame takes:                                         {:.4f} secs".format(elap_estimate))
    print("Estimated total time to finish (detection at every frame, no tracking):  {:.2f} secs".format(elap_estimate*total))
    print("Actual Elapsed time with tracking:                                       {:.2f} secs".format(fps.elapsed()))
    print("Approx. FPS:                                                             {:.2f}".format(fps.fps()))

    df = pd.DataFrame(np.matrix(df), columns = ['frame', 'ID','start_x','start_y', 'end_x', 'end_y', 'centroid_x', 'centroid_y'])
    df.to_csv('tracked.csv')
    df = pd.read_csv('tracked.csv')
    df.describe()
    df.head()
    print("Number of unique objects detected and tracked:                          ", len(df['ID'].unique()))

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
    if writer_orig is not None or writer_thresh is not None or writer_bilat is not None:
        writer_orig.release()
        writer_thresh.release()
        writer_bilat.release()

    # close any open windows
    cv2.destroyAllWindows()


### To compare videos generated from different kind of pre-prcoessing for analysis
def compare():
    orig_path = os.path.join(args.output_path, 'out_orig.mp4')
    # gray_path = os.path.join(args.output_path, 'out_cow_gray.mp4')
    thresh_path = os.path.join(args.output_path, 'out_thresh.mp4')

    vs_orig = cv2.VideoCapture(orig_path)
    # vs_gray = cv2.VideoCapture(gray_path)
    vs_thresh = cv2.VideoCapture(thresh_path)

    while(True):
        frame_orig = vs_orig.read()
        # frame_gray = vs_gray.read()
        frame_thresh = vs_thresh.read()

        frame_orig = frame_orig[1]
        frame_thresh = frame_thresh[1]
        # frame_gray = frame_gray[1]

        if frame_orig is None or frame_thresh is None:
                break

        cv2.imshow("Orig", frame_orig)
        # cv2.imshow("Gray", frame_gray)
        cv2.imshow("Thresh", frame_thresh)
        time.sleep(0.3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs_orig.release()
    # vs_gray.release()
    vs_thresh.release()
    cv2.destroyAllWindows()


### Generate config yaml file for arguments
def generate_yaml(name):

    data = {'model' : 1,
            'model_path' :'./model_checkpoint',
            'input_type' : 0,
            'data_path' : '../data/cow5/',
            'output_path' : './output_checkpoints/',
            'fps' : 4.5,
            'max_disappeared' : 10,
            'max_distance' : 200,
            'resize' : True,
            'im_width' : 700,
            'confidence' : 0.01,
            'thresh' : 0.3,
            'skip_frames' : 6,
            'filter_type' : 1,
            'sigma_color' : 140,
            'sigma_space' : 100,
            'diam' : 27,
            'block_size' : 21,
            'constant' : 5}

    f = open(name, 'w')
    yaml.dump(data)
    yaml.dump(data, f)
    f.close()



if __name__ == '__main__':
    p = configargparse.ArgParser()

    p.add('-c', '--config', required=False, is_config_file=True, help='Script running Autonomous object detection and tracking on cow5 data',
          default = '../configs/cow6_best.yml')
    # Required Paths
    p.add("--model", type = int, help = "Choose 0 for MobileNet and 1 for YOLOv3", default = 1)
    p.add("--model_path", type = str, help="path to Caffe pre-trained model", default = './model_checkpoint')
    p.add("--input_type", type=int, help="Choose 0 for image sequences and 1 for video", default = 0)
    p.add("--data_path", type=str, help="path to input video file", default = '../data/cow6/')
    p.add("--output_path", type=str, help="path to optional output video file", default = './output_checkpoints/')
    p.add("--fps", type = float, help = "At what fps to write to output file for playback analysis", default = 3)

    # Pre-prcoessing parameters
    ## Filter parameters
    p.add("--filter_type", type = int, help = "Choose 0 for Gaussian or 1 for BiLateral filter", default = 1)
    p.add("--sigma_color", type = int, help = "Filter sigma in the color space (ideally between 10-150)", default = 50)
    p.add("--sigma_space", type = int, help = "Filter sigma in the co-ordinate space (ideally between 10-150)", default = 50)
    p.add("--diam", type = int, help = "Diameter of neighborhood", default = 15)
    ## Thresholding parameters
    p.add("--block_size", type = int, help = "Pixel neighbourhood size for adaptive thresholding", default = 9)
    p.add("--constant", type = int, help = "Constant subtracted from weighted mean in adaptive thresholding", default = 5)

    # Detection and Tracking Parameters
    p.add("--max_disappeared", type=int, help = "Maximum frames a tracked object can be marked as 'disappeared' before forgetting it",
          default = 4)
    p.add("--max_distance", type=int, help ="Maximum distance the object should have drifted from its previous position to mark it is disappeared and treat it as a new object",
          default = 200)
    p.add("--resize", type = bool, help = "Whether to resize the frames for faster processing", default = True)
    p.add("--im_width", type = int, help = "Image Width for resizing the image", default = 700)
    p.add("--confidence", type=float, help="minimum probability to filter weak detections", default = 0.01)
    p.add("--thresh", type =float, help = "threshold when applying non-maximum suppression", default = 0.3)
    p.add("--skip_frames", type=int, help="No. of frames to skip between detections", default= 5)

    args = p.parse_args()
    print("\n"+"-"*50 + "\n\t\tLoaded arguments\n" + "-"*50 + "\n")
    for arg in vars(args):
        print("{} :  {}".format(arg, getattr(args, arg)))

    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]


    # Assert the requried paths and correct combination of arguments
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
        model_path = os.path.join(args.model_path, 'MobileNetSSD_deploy.caffemodel')#MobileNetSSD_deploy
        prototxt = os.path.join(args.model_path, 'MobileNetSSD_deploy.prototxt')#'MobileNetSSD_deploy.prototxt')
    elif args.model == 1:
        wt_path  = os.path.join(args.model_path, 'yolov3.weights')
        config_path = os.path.join(args.model_path, 'yolov3.cfg')
    else:
        sys.exit("[!] WARNING !! Incorrect Model selection: Choose 0 for MobileNet and 1 for YOLOv3")

    # load our serialized model from disk
    print("\n"+"="*80 + "\n\t\t\t\t TRACKING\n" + "="*80)
    print("\nLoading model...")

    if args.model == 0:
        model = cv2.dnn.readNetFromCaffe(prototxt, args.model_path)
        layer_names = None
    else:
        model = cv2.dnn.readNetFromDarknet(config_path, wt_path)
        layer_names = model.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    if not os.path.exists(args.output_path):
        print("\nCreating checkpoint path for output videos:  ", checkpoint_path)
        os.makedirs(checkpoint_path)
    else:
        print("\nOutput videos written at: ", args.output_path)

    if not (args.input_type==0 or args.input_type == 1 ):
        sys.exit("[!] Incorrect Input Type argument: Choose 0 for image sequences and 1 for video")

    Main_Handler(args, model, layer_names, args.data_path, args.output_path)
    # compare()
    # generate_yaml(name = '../configs/cow5_best.yml')


    #################
    # Best Settings #
    #################

    # brain2.mp4 : 100, 75, 0.65, 0.55, 15 with YOLOv3
    # cow5/  :  10,1000, 0, 0.25, 10, bilat = 35,150,100, adathresh 21,10