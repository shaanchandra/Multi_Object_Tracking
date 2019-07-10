# Multi_Object_Tracking
Multi Object detection and tracking in videos 

## Overview

The repository contains object tracking using 2 methods:

1. **Manual custom BoundingBox + tracking**: This method is implemented under `Custom_Tracking`. The program outputs each frame when run, allowing to be paused at any time and mark the object of interest by dragging a bounding-box across the frame using the cursor. In all the subsequent frames, the program uses one of the 5 specified tracking algorithms implemented in OpenCV (*klf, mosse, medianflow, ...*) to track the object.

2. **Autonomous detection and tracking with unique ID tags**: This method is implented under `Autonomous_Tracking`. 

Both the programs generate the processed video with the tracked BoundingBox markings, a track plot and heat map of all the tracked objects, a `.csv` file with the *x* and *y* co-ordinates of each uniquely tracked object in each frame for any form of further processing (for instance, behavorial analysis, clustering, etc).

## 




