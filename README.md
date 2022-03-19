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
      3.main.py -- Starting with Loading the data, Implementing the detection, Tracking, Prediction, and Update  
      4.project_video.mp4 -- Video of a vehicle travelling on the road which is split into frames.
 

## DATA SOURCE

Originial video:
https://github.com/kcg2015/Vehicle-Detection-and-Tracking/blob/master/project_video.mp4
Video captured into frames:
https://github.com/kcg2015/Vehicle-Detection-and-Tracking/tree/master/test_images

## STATE EQUATIONS

The general linear dynamic system's state equation used is of the form:  𝑥𝑘=𝑥𝑘−1+𝑢𝑘−1Δ𝑡+0.5∗𝑎𝑘−1∗Δ𝑡2 
where,
𝑥𝑘  = state at a given time stamp 'k' (current state)
𝑥𝑘−1  = = state at a given time 'k-1' (prior state)
Δ𝑡  is change in time
u is the velocity at time 'k-1' (which is controlling the satates)
a is acceleration at time 'k-1'
In this analysis, we consider the object travelling in 2-D (both in X and Y direction).
Hence there will be state equations in terms of x and y.
Here, we are considering the four corners of the bounding box.


!!!!!! bounding image


In a X-Y plane:

y1 = y2 ; y4 = y3 (since the pair of points are on same horizontal line)
x1 = x4 ; x2 = x3 (since the pair of points are on same verticle line)

Hence, there will be four state equtions:
𝑥1,𝑘=𝑥1,𝑘−1+𝑢1,𝑘−1Δ𝑡 
𝑥2,𝑘=𝑥2,𝑘−1+𝑢2,𝑘−1Δ𝑡 
𝑦1,𝑘=𝑦1,𝑘−1+𝑢1,𝑘−1Δ𝑡 
𝑦1,𝑘=𝑦1,𝑘−1+𝑢1,𝑘−1Δ𝑡 

(As already mentioned acceletaion term is zero, since velocity os considered constant.)


The state equation are always paired wih measurement equations, that describes the relationship between state and measurement at the current time stamp k.
The corresponding equations are obtained by taking the first derivative of staes with respect to time:
𝑧𝑥1,𝑘=𝑢𝑥1,𝑘−1 
𝑧𝑥2,𝑘=𝑢𝑥2,𝑘−1 
𝑧𝑦1,𝑘=𝑢𝑦1,𝑘−1 
𝑧𝑦2,𝑘=𝑢𝑦2,𝑘−1 


## Notations:

X - State Mean
P - State Covariance
F - State Transition Matrix
Q - Process Covariance
B - Control Function
u - Control Input
Here, Q consists of the variances associated with each of the state estimates as well as the correlation between the errors in the state estimates. 

![alt text](https://github.com/Karthika-ai/Vehicle-Detection-and-Tracking-Using-Kalman-Filter/blob/main/Screenshots/Screen%20Shot%202022-03-19%20at%208.36.58%20PM.png?raw=true)















![alt text](https://github.com/kcg2015/Vehicle-Detection-and-Tracking/raw/master/example_imgs/low_meas_noise.png)


In contrast, if measurement noise is high, the updated state is very close to the initial prediction (aqua bounding box completely overlaps over the green bounding box).

![alt text](https://github.com/kcg2015/Vehicle-Detection-and-Tracking/raw/master/example_imgs/high_meas_noise.png)



## Detection-to-Tracker Assignment

## Issues

The main issue is occlusion. For example, when one car is passing another car, the two cars can be very close to each other. This can fool the detector into outputing a single (and possibly bigger bounding) box, instead of two separate bounding boxes. In addition, the tracking algorithm may treat this detection as a new detection and sets up a new track. The tracking algorithm may fail again when one of the passing car moves away from another.
