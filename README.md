# Multi_Object_Tracking
Multiple Object detection and tracking in videos 

## Overview

The repository contains object tracking using 2 methods:

1. **Custom BoundingBox + tracking**: This method is implemented under `Custom_Tracking`. The program outputs each frame when run, allowing to be paused at any time and mark the object of interest by dragging a bounding-box across the frame using the cursor. In all the subsequent frames, the program uses one of the 5 specified tracking algorithms implemented in OpenCV (*csrt, kcf, boosting, mil, tild, mosse and medianflow*) to track the object. There is also an option to choose a more advanced tracker called the Distractor-aware Siamese Region Propsal Network (DaSiamRPN).

2. **Autonomous detection and tracking with unique ID tags**: This method is implented under `Autonomous_Tracking`. The program offers the option to either use pre-trained `YOLOv3` or `MobileNetSSD`(claimed to be 10 times faster in the original paper, I get it to be performing 4-5 times faster but with slightly poor qualitative performance) model for autonomous object detection in frames. I have used *Correlation_filter* of `dlib` and `centroid tracking` for tracking all the detected objects over the frames. Detection is run every few frames (set by a parameter) and in between detections only tracking phase operates. Objects leaving the frame or 'disappearing' can be dealt with a parameter too (No. of successive frames of failed object tracking before marking it as 'absent' and clearing its ID tracker).

Both the programs generate the processed video with the tracked BoundingBox markings, a track plot and heat map of all the tracked objects and a `.csv` file with the *x* and *y* co-ordinates of each uniquely tracked object in each frame to aid any form of further processing (for instance, behavorial analysis, clustering, etc).

Both the codes can run either on video files (ie, *mp4* or *avi* files), or a sequence of image frames stored in a folder as many datasets or test cases come in both formats.

## Code

It is advisable to maintain the hierarchy of folders as present here. However, if you change the structure of code make sure to change file paths accordingly.

### Custom Tracking

#### ```simple_tracker.py```

This is the main file to run the custom tracking algorithm. Run this file with the specified arguments and the frame window will pop-up with the video streaming. Press ***'s'*** at any instance when you want to pause the video streaming and then drag a bounding box using the cursor to mark the *Region of Interest*. Press *ENTER* or *SPACE* to start tracking the marked object. Press ***'q'*** to exit tracking at any point. The program accepts the following command line arguments:

1. ***vid_path***(str): path to input video file/image sequences (NOTE: in case of video file specify the video file (ending in *.mp4/.avi*) and in case of image sequences just the folder that contains them.
2. ***input_type***(int): Choose 0 for image sequences and 1 for video file
3. ***model***(int): Choose 0 for CV2 tracker and 1 for DaSiamRPN tracker
4. ***multi***(bool): Choose True for Multi object tracking and False for single object
5. ***tracker***(str): OpenCV object tracker name

The program will output a *mp4* file with the marked bounding boxes in each frame, a *csv* file with the marked centroid co-ordinates of each unique object in each frame and a *png* file containing the traced paths of each unique objects (*x-y coordinate system*).

### Autonomous detection and Tracking

#### ```tracker_main.py```

This is the main file to run the autonomous object detection and tracking algorithm. Note that there are various tunable parameters that might have to be set by trial and error for different use cases. These are listed below along with the command line arguments:

*Paths and selecteion arguments:*

1. ***model***(int): Choose 0 for MobileNetSSD and 1 for YOLOv3 as the object detection model.
2. ***prototext***(str): Path to Caffe 'deploy' prototxt file. Required if model = 0.
3. ***model_path***(str): Path to the 'model_checkpoints' folder. The prrogram will choose to read the relevant model file automatically based on the selection of `model` parameter.
4. ***input_type***(int): Choose 0 if input is image sequences (video frames) and 1 if it is a video file
5. ***data_path***(str): path to input video file/image sequences (NOTE: in case of video file specify the video file (ending in *.mp4/.avi*) and in case of image sequences just the folder that contains them.
6. ***output_path***(str): Path to the folder where you want to save the processed output file.

*Tuning parameters:*

7. ***max_disappeared***(int): Maximum subsequent frames a tracked object can be marked as 'disappeared' before forgetting it.
8. ***max_distance***(int): Maximum distance the object should have drifted from its previous position to mark it is disappeared and treat it as a new object from now on
9. ***resize***(bool): Resize the frames for faster processing
10. ***im_width***(int): If resize=True then specify the frame width of the resized image
11. ***confidence***(int): minimum probability to filter weak detections
12. ***thresh***(int): threshold when applying non-maximum suppression (overalp between the RPN boxes allowed to be considered as separate objects)
13. ***skip_frames***(int): No. of frames to skip between detections and tracking phases





