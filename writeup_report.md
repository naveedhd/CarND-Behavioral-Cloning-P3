# Self-Driving Car Engineer Nanodegree

---

## Behavioral Cloning Project 

---


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* output_video.mp4 showing how model runs on the simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started with [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, and the car drove the complete first track after just three training epochs. Then I modified the model a bit to avoid overfitting. This model can be found [here](model.py#L109-L133).

#### 2. Attempts to reduce overfitting in the model

I added Dropouts between layers and L2 regularizers to convolutional layers to help avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py#L154]([model.py#L154)). Batch size for generator was chosen to be 16.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Also, the data provided by Udacity, I used the first track and second track data. The simulator provides three different images: center, left and right cameras. Each image was used to train the model.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with  [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The only modification was to add a new layer at the end to have a single output as it was required.

I then played around with adding/removing layers and changing batch_size and epochs. This did not improved the performance of driving and also took much longer to train.

I then reverted to original model and added dropouts and regularizers to avoid overfitting.

With the final model, vehicle is able to do the first track.

#### 2. Final Model Architecture

The final model architecture:
A model summary is as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
_________________________________________________________________
dropout_1 (Dropout)          (None, 43, 158, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
dropout_2 (Dropout)          (None, 20, 77, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 37, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 35, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
dropout_5 (Dropout)          (None, 4, 33, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dropout_6 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_7 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dropout_8 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 981,819.0
Trainable params: 981,819.0
Non-trainable params: 0.0
```

#### 3. Creation of the Training Set & Training Process

I used Udacity training data, as it was too hard for me to drive the vehicle correctly in the simulator.

For preprocessing the data, I used Lambda layer to perform normilizing and cropping the input images.
