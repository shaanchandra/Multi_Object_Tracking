from imutils.video import VideoStream
from imutils.video import FPS

import pandas as pd
import numpy as np
import helpers as utils
import argparse
import imutils
import time
import dlib
import cv2
import glob
import datetime

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
import os

mot = True

# def process(args, frame, meas_last, meas_now, colors, ids, writer, df, this):
#     fourcc = cv2.VideoWriter_fourcc(*args.codec)
#     (H, W) = frame.shape[:2]
#     writer = cv2.VideoWriter(checkpoint_path + 'out.mp4', fourcc, 30, (W, H), True)
#     kernel = np.ones((5,5),np.uint8)
#     # Preprocess the image for background subtraction
#     frame = cv2.resize(frame, None, fx = args.scaling, fy = args.scaling, interpolation = cv2.INTER_LINEAR)
#     # Apply mask to aarea of interest
#     # mask = np.zeros(frame.shape)
#     # mask = cv2.rectangle(mask, (100, 30), (750,585), (255,255,255), -1)
#     # frame[mask ==  0] = 0
#     thresh = utils.colour_to_thresh(frame, args.block_size, args.offset)
#     # thresh = cv2.erode(thresh, kernel, iterations = 1)
#     # thresh = cv2.dilate(thresh, kernel, iterations = 1)
#     final, contours, meas_last, meas_now = utils.detect_and_draw_contours(frame, thresh, meas_last, meas_now, args.min_area, args.max_area)
#     if len(meas_now) != args.n_inds:
#         contours, meas_now = utils.apply_k_means(contours, args.n_inds, meas_now)
    
#     row_ind, col_ind = utils.hungarian_algorithm(meas_last, meas_now)
#     final, meas_now, df = utils.reorder_and_draw(final, colors, args.n_inds, col_ind, meas_now, df, mot, this)
    
#     # Create output dataframe
#     for i in range(args.n_inds):
#         df.append([this, meas_now[i][0], meas_now[i][1], ids[i]])
    
#     # Display the resulting frame
#     writer.write(final)
#     cv2.imshow('frame', final)

#     return meas_last, meas_now, writer



# def Main_Handler(args, path, checkpoint_path):
#     ## Individual location(s) measured in the last and current step
#     meas_last = list(np.zeros((args.n_inds,2)))
#     meas_now = list(np.zeros((args.n_inds,2)))
#     df = []
#     writer = None

#     colors, ids = [], []
#     [colors.append((255,0,0)) for i in range(args.n_inds)]
#     [ids.append(str(i)) for i in range(args.n_inds)]

#     if args.input_type == 1:
#         vs = cv2.VideoCapture(path)
#         if vs.isOpened() == False:
#             sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

#         try:
#             prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
#             total = int(vs.get(prop))
#         except:
#             print("[!] WARNING ! Could not determine the No. of frames in the video. Can not estimate completion time")

#         while(True):
#             frame = vs.read()
#             frame = frame[1]
#             this = vs.get(1)

#             # if we are viewing a video and we did not grab a frame then we have reached the end of the video
#             if frame is None:
#                 break

#             meas_last, meas_now, writer = process(args, frame, meas_last, meas_now, colors, ids, writer, df, this)

#             key = cv2.waitKey(1) & 0xFF
#             # if the `q` key was pressed, break from the loop
#             if key == ord("q"):
#                 break


#     print("\n" + "-"*50 + "\n\t\tSTATISTICS:\n" + "-"*50)
#     print("\nTotal frames in video being processed:                                  ", total)
#     # print("Actual Elapsed time with tracking:                                       {:.2f} secs".format(fps.elapsed()))
#     # print("Approx. FPS:                                                             {:.2f}".format(fps.fps()))


#     # df = pd.DataFrame(np.matrix(df), columns = ['frame', 'ID','start_x','start_y', 'end_x', 'end_y', 'centroid_x', 'centroid_y'])
#     # df.to_csv('tracked.csv')
#     # df = pd.read_csv('tracked.csv')
#     # df.describe()
#     # df.head()
#     # print("Number of unique objects detected and tracked:                          ", len(df['ID'].unique()))

#     # for idx, ID in enumerate(np.unique(df['ID'])):
#     #     df['ID'][df['ID'] == ID] = idx

#     # plt.figure(figsize=(8,8))
#     # plt.scatter(df['centroid_x'], df['centroid_y'], c=df['ID'], cmap='jet')
#     # plt.xlabel('X', fontsize=16)
#     # plt.ylabel('Y', fontsize=16)
#     # plt.tight_layout()
#     # plt.savefig('tracked_vis.png', format='png', dpi=300)
#     # plt.show()

#     # check to see if we need to release the video writer pointer
#     if writer is not None:
#         writer.release()

#     # close any open windows
#     cv2.destroyAllWindows()



def Main_Handler(args, path, checkpoint_path):
    ## Individual location(s) measured in the last and current step
    meas_last = list(np.zeros((args.n_inds,2)))
    meas_now = list(np.zeros((args.n_inds,2)))
    df = []
    writer = None
    firstFrame = None

    colors, ids = [], []
    [colors.append((255,0,0)) for i in range(args.n_inds)]
    [ids.append(str(i)) for i in range(args.n_inds)]

    if args.input_type == 1:
        vs = cv2.VideoCapture(path)
        if vs.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')

        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
        except:
            print("[!] WARNING ! Could not determine the No. of frames in the video. Can not estimate completion time")
        count=0
        while(True):
            frame = vs.read()
            text = "Waiting"
            frame = frame[1]

            count+=1
            if count<1200:
                continue

            this = vs.get(1)

            # if we are viewing a video and we did not grab a frame then we have reached the end of the video
            if frame is None:
                break

            fourcc = cv2.VideoWriter_fourcc(*args.codec)
            (H, W) = frame.shape[:2]
            writer = cv2.VideoWriter(checkpoint_path + 'out.mp4', fourcc, 30, (W, H), True)

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
         
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and first frame
            # backSub = cv2.createBackgroundSubtractorMOG2()
            # backsub = cv2.createBackgroundSubtractorKNN()
            # frameDelta = backSub.apply(gray)

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
         
            # dilate the thresholded image to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
         
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args.min_area or cv2.contourArea(c) > args.max_area:
                    continue
         
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                # rects.append((x,y,w,h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Tracking"
            # objects = ct.update(rects)
            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
         
            # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            time.sleep(0.005)
            key = cv2.waitKey(1) & 0xFF
         
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break
        vs.release()
    else:
        path = os.path.join(path, '*.jpg')
        image_files = sorted(glob.glob(path))
        # loop over frames from the video stream
        for image in image_files:
            frame = cv2.imread(image)
            text = "Waiting"

            # if we are viewing a video and we did not grab a frame then we have reached the end of the video
            if frame is None:
                break

            fourcc = cv2.VideoWriter_fourcc(*args.codec)
            (H, W) = frame.shape[:2]
            writer = cv2.VideoWriter(checkpoint_path + 'out.mp4', fourcc, 30, (W, H), True)

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
         
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # thresh = cv2.threshold(frameDelta,)

            # dilate the thresholded image to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
         
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args.min_area or cv2.contourArea(c) > args.max_area:
                    continue
         
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Tracking"

            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
         
            # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            time.sleep(2)
            key = cv2.waitKey(1) & 0xFF
         
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break


    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument("--data_path", type=str, help="path to input video file", default = '../data/tadpole_video.avi')
    parser.add_argument("--input_type", type=int, help="Choose 0 for image sequences and 1 for video", default = 1)
    parser.add_argument("--output_path", type=str, help="path to optional output video file", default = './output_checkpoints/')
    parser.add_argument("--codec", type = str, help = "Codec for videos", default = "MJPG")

    # Tuning Parameters
    parser.add_argument("--n_inds", type=int, help= "No. of expected objects in the frames (constant)", default = 1)
    parser.add_argument("--block_size", type=int, help = "Block size that defines the neighborhood size for adaptive thresholding. NOTE: should be odd ", default = 81)
    parser.add_argument("--offset", type = int, help = "Offset for adaptive thresholding. Determines the threshold relative to the neighborhood mean", default = 38)
    parser.add_argument("--scaling", type = float, help = "used to speed up tracking if video resolution is too high (use value 0-1)", default = 1)
    parser.add_argument("--min_area", type=int, help = "Minimum area of the expected object to filter noise", default = 500)
    parser.add_argument("--max_area", type=int, help = "Max area of the expected object to filter noise", default = 4000)
    args = parser.parse_known_args()[0]

    # Assert the requried paths and correct combination of arguments
    if not os.path.exists(args.data_path):
        sys.exit("[!] WARNING !! Data(input) path does NOT exist!!")
    if (args.data_path.endswith("mp4") or args.data_path.endswith("avi")) and args.input_type == 0:
        sys.exit("[!] WARNING !! Video file provided but processing mode is IMAGES")
    if args.data_path.endswith("/")  and args.input_type == 1:
        sys.exit("[!] WARNING !! Image file provided but processing mode is VIDEO")

    # load our serialized model from disk
    print("="*80 + "\n\t\t\t\t TRACKING\n" + "="*80)

    path = args.data_path
    checkpoint_path = args.output_path

    if not os.path.exists(checkpoint_path):
        print("Creating checkpoint path for output videos:  ", checkpoint_path)
        os.makedirs(checkpoint_path)
    else:
        print("Output videos written at: ", checkpoint_path)

    if not (args.input_type==0 or args.input_type == 1 ):
        sys.exit("[!] Incorrect Input Type argument: Choose 0 for image sequences and 1 for video")

    Main_Handler(args, path, checkpoint_path)

    #################
    # Best Settings #
    #################

    # mouse: 900, 2000
    # example_01: 750, 10000
    # tadpole : 60, 400