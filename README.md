## Vehicle Detection and Tracking

This project illustrates the detection and tracking of vehicles using Kalman Filter. Following operations are performed in this analysis:

    1.Prediction of current and future location of the vehicle.
   
    2.Correcting the prediction as per the new measurements attained
   
    3.Optimizing the noise created by faulty detections.
   
In this analysis, we detect and track multiple vehicles using a camera mounted inside a selfdriving car. The images captured by the camera are taken as the input. In the image, the car is captured and a bounding box is created around the car. The model “ssd_mobilenet_v1_coco” which is pretrained on MS COCO datset is used for the detection of car.

The coordinates of all four corners of the bounding are noted and state of the vehicle is determined at a particular time. Here, the rate of change of position gives the velocity. We consider the velocity as constant in the analysis. (Hence, acceleration is zero.)

Then, we implement Kalman filter to perform two main operations:

      1.Prediction of objects current location using the previous states.
      
      2.In update, current measurements are used to correct the state.


## Key files in this repo

      1.Pre Trained Model/frozen_inference_graph.pb -- pre-trained mobilenet-coco model
      2.test_images -- Video split into frame(23), which is considered as an input
      3.Data Uploading Steps -- Steps to upload the data from the github repository into the collab notebook
      4.main.py -- Starting with Loading the data, Implementing the detection, Tracking, Prediction, and Update  
      5.project_video.mp4 -- Video of a vehicle travelling on the road which is split into frames.
      6.Vehicle detection and tracking project.ipynb - Google Collab Notebook
      
 

## DATA SOURCE

Originial video:
https://github.com/kcg2015/Vehicle-Detection-and-Tracking/blob/master/project_video.mp4

Video captured into frames:
https://github.com/kcg2015/Vehicle-Detection-and-Tracking/tree/master/test_images

## STATE EQUATIONS

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/13.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/Screen%20Shot%202022-03-19%20at%208.44.05%20PM.png?raw=true)


![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/1.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/2.png?raw=true)



## NOTATIONS

X - State Mean

P - State Covariance

F - State Transition Matrix

Q - Process Covariance

B - Control Function

u - Control Input

Here, Q consists of the variances associated with each of the state estimates as well as the correlation between the errors in the state estimates. 

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/Screen%20Shot%202022-03-19%20at%208.36.58%20PM.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/3.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/4.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/5.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/6.png?raw=true)

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/7.png?raw=true)



## IMPLEMENTATION AND TEST TRACKER

The operations - prediction and update take place in this phase.
In prediction, the thye previous statres are used to predict the current state. In update, the current measurement value (location of the bounding box) is used, to correct the state.

Here, we use the coordinates and their first-order derivatives of the up, down, left and right cornerx of the bounding box.
Assumptions:
Here, we are assuming the constant velocity (thus no acceleration).
Time interval is constant and is taken as 1

### OBSERVATION

![alt text](https://github.com/kcg2015/Vehicle-Detection-and-Tracking/raw/master/example_imgs/low_meas_noise.png)


From the above output images, we can visualize the measurement noises in the Kalman filter process.
The green bounding box gives the initial state of the car. The red bounding box gives the measurement values. The aqua colored bounding box give sthe updated state.
Here, the initial state value is 390 The measurement state value is 399 afetr 1 second Updated state value is 398
If measurement noise is low, the updated state is very close to the measurement. So, the aqua bounding box completely overlaps the red bounding box.
Suppose, if measurement noise is high, the updated state is very close to the initial prediction. So, the aqua bounding box completely overlaps over the green bounding box.
From the output images, there is no complete overlapping of green and aqua bounding box. Which infers that the measurement noise is relatively lower.


## MAIN functions

These functions, implementt the detection and tracking, including detection-track assignment and track management:

In the below logic, the section assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3) takes from current list of trackers and new detections, output matched detections, unmatched trackers, unmatched detections.

### HUNGARIAN ALGORITHM

The Hungarian method is a combinatorial optimization algorithm that solves the assignment problem in polynomial time and which anticipated later primal–dual methods. (https://en.wikipedia.org/wiki/Hungarian_algorithm)

#### EXAMPLE:

Consider a company has 4 employees and there are four jobs to be done. Here, each employee is capable of doing any of the four jobs. But, the time taken to complete the work is different for different employees. Then Hungarian Algorithm concept is applied and the jobs are assigned to the emplyees in effecient manner so as to control the time and cost of the assignment.
Here, the analysis is carried out by taking all the values of time taken by the employees for different jobs into a tablular format or in form of matrix (cost matrix) and operations are performed till we optimise the assignment.
In the cost matrix, if a constant is added to every element of a row or a column, the optimum solution of the resulting assignment problem is the same as the original problem.

In this analysis, we implement Linear Assignment and Hungarian (Munkres) algorithm:
If multiple detections are identified, we need to assign each of them to a tracker. Here, we are using intersection over union (IOU) of a 'tracker bounding box' and 'detection bounding box' as a metric. Here, we analyse till maximizing/optimizing the sum of IOU assignment.
In the below logic, linear_assignment by default minimizes an objective function. So we need to reverse the sign of IOU_mat for maximization.

## Unmatched detections and trackers

Depending on the results of linear assignment, we make lists for unmatched detections and unmatched trackers.
When a car enters into a frame and is first detected, it is not matched with any existing tracks. Hence this particular detection is categorized under unmatched detection. If any matching with an overlap less than iou_thrd, it denotes the existence of an untracked object. When the car leaves the frame, the previous used track has no more detections to consider. So, the track is considered as an unmatched track.
In this way, the tracker and the detection associated in the matching are added to the lists of unmatched trackers and unmatched detection.
We include two important design parameters, min_hits and max_age, in the pipeline. The parameter min_hits is the number of consecutive matches needed to establish a track. The parameter max_age is number of consecutive unmatched detections before a track is deleted. Both parameters need to be tuned to improve the tracking and detection performance.


![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/8.png?raw=true)
![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/9.png?raw=true)
![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/10.png?raw=true)
![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/11.png?raw=true)
![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/12.png?raw=true)



## CONCLUSION:

From all the above output images, the condition trk.hits >= the number of consecutive matches need for track establishment and trk.no_losses <= the number of cosecutive unmatched detections before a track is deleted. Hence the track is fully established.
As the result, the bounding box is created in the output image, as shown in the above figure.
So, we have implemented the detection and tracking logic which includes detection track assignment and track management. In this analysis, the detecter first localized the vehicles in each frame. Then the tracker is updated with detected results. Hence we predicted and updated the location based on the current state. Or it can also be said as we have predicted the cars next position depending upon its previous position. We have also dealt with noise measurements by looking at how the boxes ovelap on eachother depending on the noise levels.

The model has performed pretty well and produced desirable results in prediction, measurement and updation, depending the assumptions taken into consideration.


## CITATION:

1. The logic in the work https://github.com/kcg2015/Vehicle-Detection-and-Tracking is taken as a reference for this analysis.
2. https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
3. https://machinelearningspace.com/object-tracking-python/




