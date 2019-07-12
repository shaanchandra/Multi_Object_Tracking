from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import glob
import time
import cv2
import torch

import sys
sys.path.append("..")

import os
import matplotlib.pyplot as plt
import numpy as np
from os.path import realpath, dirname, join
import pandas as pd

from siam_tracker.net import SiamRPNvot
from siam_tracker.run_SiamRPN import SiamRPN_init, SiamRPN_track
from siam_tracker.utils import get_axis_aligned_bbox, cxy_wh_2_rect



def CV2_TRACKER(frame, tracker, initBB, fps, succcess_count, fail_count, df):
    (H, W) = frame.shape[:2]

    # grab the new bounding box coordinates of the object
    (success, box) = tracker.update(frame)

    if success:
        succcess_count+=1
    else:
        fail_count +=1

    # check to see if the tracking was a success
    if success:
        if not args.multi:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
            df.append([x,y])
        else:
            for b in box:
                (x, y, w, h) = [int(v) for v in b]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                df.append([x,y])


    # update the FPS counter
    fps.update()
    fps.stop()

    # initialize the set of information we'll be displaying on
    # the frame
    info = [
        ("Tracker", args.tracker),
        ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    return initBB, fps, succcess_count, fail_count, df



def Siam_TRACKER(frame, tracker, initBB, state, fps, df):
    (H, W) = frame.shape[:2]

    # tracking and visualization
    state = SiamRPN_track(state, frame)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    df.append([res[0], res[1]])

    # update the FPS counter
    fps.update()
    fps.stop()
    res = [int(l) for l in res]
    cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 0), 2)
    cv2.putText(frame, str(fps.fps()), (10, H - ((1 * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.waitKey(1)

    return frame, tracker, initBB, state, fps, df


def Main_Handler(args, tracker, path):
    print("="*80 + "\n\t\t\t\t TRACKING\n" + "="*80)

    if args.input_type == 0:

        # initialize the bounding box coordinates of the object we are going to track
        initBB = None
        # initialize the FPS throughput estimator
        fps = None
        succcess_count = 0
        fail_count = 0
        df = []
        path = os.path.join(path, '*.jpg')
        image_files = sorted(glob.glob(path))
        # loop over frames from the video stream
        for image in image_files:
            frame = cv2.imread(image)

            # resize the frame (so we can process it faster) and grab the frame dimensions
            frame = imutils.resize(frame, width=800)

            if initBB is not None:
                if args.model == 0:
                    initBB, fps, succcess_count, fail_count, df = CV2_TRACKER(frame, tracker, initBB, fps, succcess_count, fail_count, df)
                else:
                    frame, tracker, initBB, state, fps, df = Siam_TRACKER(frame, tracker, initBB, state, fps, df)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(33) & 0xFF
            # if the 's' key is selected, we are going to "select" a bounding
            # box to track
            if key == ord("s"):
                # select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)

                if args.model == 1:
                    x,y,w,h = initBB
                    # tracker init
                    target_pos, target_sz = np.array([x, y]), np.array([w, h])
                    state = SiamRPN_init(frame, target_pos, target_sz, tracker)
                else:
                    # start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
                    tracker.init(frame, initBB)

                fps = FPS().start()
        
            # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break


    else:
        # initialize the bounding box coordinates of the object we are going to track
        initBB = None
        vs = cv2.VideoCapture(path)
        if vs.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file')
        # initialize the FPS throughput estimator
        fps = None
        succcess_count = 0
        fail_count = 0
        df = []
    
        # loop over frames from the video stream
        while True:
            # grab the current frame
            frame = vs.read()
            frame = frame[1]

            # resize the frame (so we can process it faster) and grab the frame dimensions
            frame = imutils.resize(frame, width=400)

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            if initBB is not None:
                if args.model == 0:
                    initBB, fps, succcess_count, fail_count, df = CV2_TRACKER(frame, tracker, initBB, fps, succcess_count, fail_count, df)
                else:
                    frame, tracker, initBB, state, fps, df = Siam_TRACKER(frame, tracker, initBB, state, fps, df)


            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(33) & 0xFF
            # if the 's' key is selected, we are going to "select" a bounding
            # box to track
            if key == ord("s"):
                # select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)

                if args.model == 1:
                    x,y,w,h = initBB
                    # tracker init
                    target_pos, target_sz = np.array([x, y]), np.array([w, h])
                    state = SiamRPN_init(frame, target_pos, target_sz, tracker)
                else:
                    if args.multi:
                        tr = OPENCV_OBJECT_TRACKERS[args.tracker]
                        tracker.add(tr, frame, initBB)
                    # start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
                    else:
                        tracker.init(frame, initBB)

                fps = FPS().start()
        
            # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break

        vs.release()


    # close all windows
    cv2.destroyAllWindows()
    print("\nStatistics: \n")
    print("Success rate: {:.2f}%".format(succcess_count/(succcess_count+fail_count+1e8) * 100))

    df = pd.DataFrame(np.matrix(df), columns = ['centroid_x', 'centroid_y'])
    df.to_csv('tracked.csv')
    df = pd.read_csv('tracked.csv')
    df.describe()
    df.head()

    # for idx, ID in enumerate(np.unique(df['ID'])):
    #     df['ID'][df['ID'] == ID] = idx

    plt.figure(figsize=(8,8))
    plt.scatter(df['centroid_x'], df['centroid_y'], cmap='jet')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.tight_layout()
    plt.savefig('tracked_vis.png', format='png', dpi=300)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", type=str, help="path to input video file", default = "../data/brain2.mp4")
    parser.add_argument("--input_type", type=int, help="Choose 0 for image sequences and 1 for video", default = 1)
    parser.add_argument("--model", type=int, help="Choose 0 for CV2 tracker and 1 for DaSiamRPN tracker", default = 0)
    parser.add_argument("--multi", type=bool, help="Multi object tracking or single object", default = True)
    parser.add_argument("--tracker", type=str, help="OpenCV object tracker type", default="kcf")
    args = parser.parse_known_args()[0]

    # (major, minor) = cv2.__version__.split(".")[:2]
    # print(major, minor)
    if not os.path.exists(args.vid_path):
        sys.exit("[!] WARNING !! Data(input) path does NOT exist!!")
    if (args.vid_path.endswith("mp4") or args.vid_path.endswith("avi")) and args.input_type == 0:
        sys.exit("[!] WARNING !! Video file provided but processing mode is IMAGES")
    if args.vid_path.endswith("jpg")  and args.input_type == 1:
        sys.exit("[!] WARNING !! Image file provided but processing mode is VIDEO")


    OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerKCF_create(),
            "kcf": cv2.TrackerKCF_create(),
            "boosting": cv2.TrackerBoosting_create(),
            "mil" : cv2.TrackerMIL_create(),
            "tld" : cv2.TrackerTLD_create(),
            "medianflow" : cv2.TrackerMedianFlow_create(),
            "mosse": cv2.TrackerMOSSE_create()
            }

    if args.multi == False:
        if args.model == 0:
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]
        elif args.model == 1:
            tracker = SiamRPNvot()
            tracker.load_state_dict(torch.load(join(realpath(dirname(__file__)), './siam_tracker/SiamRPNVOT.model')))
            tracker.eval()
        else:
            sys.exit("[!] Incorrect Model argument: Choose 0 for CV2 tracker and 1 for DaSiamRPN tracker")
    else:
        tracker = cv2.MultiTracker_create()




    path = args.vid_path

    Main_Handler(args, tracker, path)







