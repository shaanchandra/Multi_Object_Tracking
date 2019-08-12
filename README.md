# Multi_Object_Tracking
Multiple Object detection and tracking in videos 

## Overview

Each video frame/image sequence was pre-processed in primarily 2 steps before detection and tracking:

1. **Bilateral Filtering**: The bilateral filter uses 2 Gaussian filters, one in the space domain and other (multiplicative) Gaussian filter in the pixel intensity domain. The Gaussian function of space makes sure that only pixels are *‘spatial neighbors’* are considered for filtering, while the Gaussian component applied in the intensity domain (a Gaussian function of intensity differences) ensures that only those pixels with intensities similar to that of the central pixel (‘intensity neighbors’) are included to compute the blurred intensity value. As a result, this method *preserves edges*, since for pixels lying near edges, neighboring pixels placed on the other side of the edge, and therefore exhibiting large intensity variations when compared to the central pixel, will not be included for blurring. The original paper is [Bilateral Filtering for Gray and Color Images](http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Tomasi98.pdf)

2. **Adaptive Gaussian Thresholding**: We perform thresholding as a step for *foregroud-background segmentation*. Having a global threshold value is not ideal as different regions of the image can have differnt lighting conditions. Thus the adaptive threshold algorithm calculates the threshold as the Gaussian weighted sum of each local neighborhood. This gives us different thresholds for different regions of the same image.

The repository contains object tracking using 3 methods:

1. **Custom BoundingBox + tracking**: This method is implemented under `Custom_Tracking`. The program outputs each frame when run, allowing to be paused at any time and mark the object of interest by dragging a bounding-box across the frame using the cursor. In all the subsequent frames, the program uses one of the 5 specified tracking algorithms implemented in OpenCV (*csrt, kcf, boosting, mil, tild, mosse and medianflow*) to track the object. There is also an option to choose a more advanced tracker called the Distractor-aware Siamese Region Propsal Network (DaSiamRPN).

2. **Autonomous detection and tracking with unique ID tags**: This method is implemented under `Autonomous_Tracking`. The program offers the option to either use pre-trained `YOLOv3` or `MobileNetSSD`(claimed to be 10 times faster in the original paper, I get it to be performing 4-5 times faster but with slightly poor qualitative performance) model for autonomous object detection in frames. I have used *Correlation_filter* of `dlib` and `centroid tracking` for tracking all the detected objects over the frames. Detection is run every few frames (set by a parameter) and in between detections only tracking phase operates. Objects leaving the frame or 'disappearing' can be dealt with a parameter too (No. of successive frames of failed object tracking before marking it as 'absent' and clearing its ID tracker). Tracking operates in the following steps:

    1. **Tracking initialization and Unique ID taggig**: A tracker is initialized for each bounding box detected by the detection model.
    2. **Target localization using Correlation Filter**: Each detected object is initialized with a tracker that solves a regression problem in the frequency domain using *Fast Fourier Transform* to predict the correct bounding box of the object in the new frame. It does so with real-time FPS throughput.
    3. **Identity resolution using Centroid Tracker**: Computes Euclidean distance between the centroids ofeach bounding box in the new and old frames and assigns the correct IDs to the new boxes.


3. **Background Segmentation**: This method is implemented under `Back_segm`. The program uses foreground-background segmentation methods with diluion and thresholding to track moving objects. Works under the strong assumption that the first frame of the video has the background only with no objects of interest.

Both the programs generate the processed video with the tracked BoundingBox markings, a track plot and heat map of all the tracked objects and a `.csv` file with the *x* and *y* co-ordinates of each uniquely tracked object in each frame to aid any form of further processing (for instance, behavorial analysis, clustering, etc).

Both the codes can run either on video files (ie, *mp4* or *avi* files), or a sequence of image frames stored in a folder as many datasets or test cases come in both formats.

## Setting up the environment

1. I would recommend installing Anaconda. That gets us the Conda package and environment manager, which just makes life more pleasant, in my experience.
2. Run the command `conda create -n name_of_env_here python=3.7.3` to create a virtual environment for the project.
3. Then run `conda activate name_of_env_just_created` to enter that virtual environment you just created.
3. Install all the dependencies by running the command `pip install -r requirements.txt` after cloning this repository.

It is advisable to maintain the hierarchy of folders as present here. However, if you change the structure of code make sure to be consistent with path arguments too.

## Code

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
2. ***fps***(float): At what fps to write to output file for playback analysis.
3. ***model_path***(str): Path to the 'model_checkpoints' folder. The prrogram will choose to read the relevant model file(s) automatically based on the selection of `model` parameter.
4. ***input_type***(int): Choose 0 if input is image sequences (video frames) and 1 if it is a video file
5. ***data_path***(str): path to input video file/image sequences (NOTE: in case of video file specify the video file (ending in *.mp4/.avi*) and in case of image sequences just the folder that contains them.
6. ***output_path***(str): Path to the folder where you want to save the processed output file.

*Pre-prcoessing parameters:*

7. ***filter_type***(int): Choose 0 for Gaussian or 1 for BiLateral filter
8. ***sigma_color***(int): Filter sigma in the color space(ideally between 10-150)
9. ***sigma_space***(int): Filter sigma in the co-ordinate space(ideally between 10-150)
10. ***diam***(int): Diameter of the neighborhood
11. ***block_size***(int): Pixel neighborhood size for adaptive thresholding
12. ***constant***(int): Constant subtracted from weighted mean in adaptive thresholding

*Detection and Tracking parameters:*

13. ***max_disappeared***(int): Maximum subsequent frames a tracked object can be marked as 'disappeared' before forgetting it.
14. ***max_distance***(int): Maximum distance the object should have drifted from its previous position to mark it is disappeared and treat it as a new object from now on
15. ***resize***(bool): Resize the frames for faster processing
16. ***im_width***(int): If resize=True then specify the frame width of the resized image
17. ***confidence***(int): minimum probability to filter weak detections
18. ***thresh***(int): threshold when applying non-maximum suppression (overalp between the RPN boxes allowed to be considered as separate objects)
19. ***skip_frames***(int): No. of frames to skip between detections and tracking phases

## Results

**Bilateral Filter on original frame** We can see that the background is blurred reducing the noise while the edges of animals are preserved. The noisy ground background is now more or less unifrom and so is the wall at the back. In contrast, we can see that the cows are not blurred and their pixels are not mixed with the surrounding pixels at the edges. Hence, bilateral filter allows for well segmented denoising and preserving edges of objects which is a desirable property for object detection tasks.

![BilaterFilter](https://github.com/shaanchandra/Multi_Object_Tracking/blob/master/results/bilat.gif)

**Adaptive thresholding on filtered frame** We can see that the background is unfiormly segmented out and the edges of animals are sharp for the tracker to get clean inputs to work on.

![AdpativeThresholding](https://github.com/shaanchandra/Multi_Object_Tracking/blob/master/results/thresh.gif)
