# Multi_Object_Tracking
Multiple Object detection and tracking in videos 

## Overview

The repository contains object tracking using 2 methods:

1. **Manual custom BoundingBox + tracking**: This method is implemented under `Custom_Tracking`. The program outputs each frame when run, allowing to be paused at any time and mark the object of interest by dragging a bounding-box across the frame using the cursor. In all the subsequent frames, the program uses one of the 5 specified tracking algorithms implemented in OpenCV (*klf, mosse, medianflow, ...*) to track the object.

2. **Autonomous detection and tracking with unique ID tags**: This method is implented under `Autonomous_Tracking`. The program uses `MobileNetSSD` model pretrained on 21 classes (20 classes + background class) for autonomous object detection in frames. I have used *Correlation_filter* of `dlib` and `centroid tracking` for tracking all the detected objects over the frames. Detection is run every few frames (set by a parameter) and in the between detections only tracking phase operates. Objects leaving the frame or 'disappearing' can be dealt with a parameter too (No. of successive frames of failed object tracking before marking it as 'absent' and clearing its ID tracker).

Both the programs generate the processed video with the tracked BoundingBox markings, a track plot and heat map of all the tracked objects, a `.csv` file with the *x* and *y* co-ordinates of each uniquely tracked object in each frame to aid any form of further processing (for instance, behavorial analysis, clustering, etc).

## Code

It is advisable to maintain the hierarchy of folders as present here. However, if you change the structure of code make sure to change file paths accordingly.






