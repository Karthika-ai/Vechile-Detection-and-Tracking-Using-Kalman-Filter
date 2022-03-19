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
 
## Detection


## Kalman Filter for Bounding Box Measurement

We use Kalman filter for tracking objects. Kalman filter has the following important features that tracking can benefit from:

Prediction of object's future location
Correction of the prediction based on new measurements
Reduction of noise introduced by inaccurate detections
Facilitating the process of association of multiple objects to their tracks
Kalman filter consists of two steps: prediction and update. The first step uses previous states to predict the current state. The second step uses the current measurement, such as detection bounding box location , to correct the state. The formula are provided in the following:

### Kalman Filter Equation:

### Prediction phase: notations

X - State Mean

P - State Covariance

F - State Transition Matrix

Q - Process Covariance

B - Control Function

u - Control Input

The coordinates of the upper right, upper left, bottom right, bottom left corners of the bounding box are  (𝑥1,𝑦1) ,  (𝑥2,𝑦2) ,  (𝑥3,𝑦3)  and  (𝑥4,𝑦4)  respectively. Since  𝑥1  must equals to  𝑥4 ,  𝑥2  must equals to  𝑥3 ,  𝑦1  must equals to  𝑦2 , and  𝑦3  equals to  𝑦4 , the coordinates can be written as  (𝑥1,𝑦1) ,  (𝑥2,𝑦1) ,  (𝑥2,𝑦3)  and  (𝑥1,𝑦3) . Changing the notation by letting  𝑦2  denotes  𝑦3 , we have  (𝑥1,𝑦1) ,  (𝑥2,𝑦1) ,  (𝑥2,𝑦2)  and  (𝑥1,𝑦2)  for upper right, upper left, bottom right, bottom left respectively.

By the assumption that velocity is constant so that accelaration is zero, we write out the following equations for each corner (state quation, control equation):

(𝑥1,𝑦1)=(𝑥1, 𝑘−1+𝑑𝑥1, 𝑘−1Δ𝑡, 𝑦1, 𝑘−1+𝑑𝑦1, 𝑘−1) 

(𝑥2,𝑦1)=(𝑥2, 𝑘−1+𝑑𝑥2, 𝑘−1Δ𝑡, 𝑦1, 𝑘−1+𝑑𝑦1, 𝑘−1) 

(𝑥2,𝑦2)=(𝑥2, 𝑘−1+𝑑𝑥2, 𝑘−1Δ𝑡, 𝑦2, 𝑘−1+𝑑𝑦2, 𝑘−1)

(𝑥1,𝑦2)=(𝑥1, 𝑘−1+𝑑𝑥1, 𝑘−1Δ𝑡, 𝑦2, 𝑘−1+𝑑𝑦2, 𝑘−1) 



### Prediction phase: equations

### Update phase: notations

### Update phase: equations

### Kalman Filter Implementation

The state vector has eight elements as follows:
```
[up, up_dot, left, left_dot, down, down_dot, right, right_dot]
```
That is, we use the coordinates and their first-order derivatives of the up left corner and lower right corner of the bounding box.

The process matrix, assuming the constant velocity (thus no acceleration), is:

```
self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                    [0, 1,  0,  0,  0,  0,  0, 0],
                    [0, 0,  1,  self.dt, 0,  0,  0, 0],
                    [0, 0,  0,  1,  0,  0,  0, 0],
                    [0, 0,  0,  0,  1,  self.dt, 0, 0],
                    [0, 0,  0,  0,  0,  1,  0, 0],
                    [0, 0,  0,  0,  0,  0,  1, self.dt],
                    [0, 0,  0,  0,  0,  0,  0,  1]])
```
		    
The measurement matrix, given that the detector only outputs the coordindate (not velocity), is:
```
self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 1, 0]])
```
The state, process, and measurement noises are :
```
 # Initialize the state covariance

 self.L = 100.0
 self.P = np.diag(self.L*np.ones(8))
       
        
 # Initialize the process covariance
 self.Q_comp_mat = np.array([[self.dt**4/2., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
 self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
# Initialize the measurement covariance
self.R_scaler = 1.0/16.0
self.R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
self.R = np.diag(self.R_diag_array)
```
Here self.R_scaler represents the "magnitude" of measurement noise relative to state noise. A low self.R_scaler indicates a more reliable measurement. The following figures visualize the impact of measurement noise to the Kalman filter process. The green bounding box represents the prediction (initial) state. The red bounding box represents the measurement. If measurement noise is low, the updated state (aqua colored bounding box) is very close to the measurement (aqua bounding box completely overlaps over the red bounding box).

![alt text](https://github.com/kcg2015/Vehicle-Detection-and-Tracking/raw/master/example_imgs/low_meas_noise.png)


In contrast, if measurement noise is high, the updated state is very close to the initial prediction (aqua bounding box completely overlaps over the green bounding box).

![alt text](https://github.com/kcg2015/Vehicle-Detection-and-Tracking/raw/master/example_imgs/high_meas_noise.png)



## Detection-to-Tracker Assignment

## Issues

The main issue is occlusion. For example, when one car is passing another car, the two cars can be very close to each other. This can fool the detector into outputing a single (and possibly bigger bounding) box, instead of two separate bounding boxes. In addition, the tracking algorithm may treat this detection as a new detection and sets up a new track. The tracking algorithm may fail again when one of the passing car moves away from another.
