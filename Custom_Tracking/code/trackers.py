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



def CV2_TRACKER(args, frame, tracker, initBB, fps, succcess_count, fail_count, df):
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

    # initialize the set of information we'll be displaying on the frame
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



def Siam_TRACKER(args, frame, tracker, initBB, state, fps, df):
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