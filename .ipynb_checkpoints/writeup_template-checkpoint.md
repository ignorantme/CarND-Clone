#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/2.jpg
[image3]: ./examples/3.jpg
[image4]: ./examples/4.jpg
[image5]: ./examples/5.jpg
[image6]: ./examples/6.jpg
[image7]: ./examples/7.jpg
[image8]: ./examples/8.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I've tried LeNet and nVidia model, and eventually chose the nVidea model, because it works obviously better. 
* First I use a Lambda layer to normalize the input images with pixel / 255.0 - 0.5, so the values fall in to the range [-0.5,0.5].
* Second a Croping layer to cut the upper 60 pixels and lower 20 pixels, leaving the area we are interested in.
* 3rd to 5th layers are convolution layers with depth of 24,36,48, all using 5x5 filter and RELU activation.
* 6th and 7th layers are 3x3 convolution layers with depth of 64 and RELU activation.
* The 8th layer is a flatten layer.
* Then a dropout layer with 0.5 keep probability.
* Then three fully connected layer with RELU activation.
* A dorpout layer with 0.5 keep probability.
* Fully connected layer with single output as the steering angle.




####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting, the data was split to 80:20, 20% for validation.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, clockwise driving, and flipped images also used to train the model.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy


####1. Solution Design Approach

As the first step I implemented the LeNet, but the result is never good, the car always drive out of the track. Then I use nVidia model mentioned on the course. It works better, at least can dirve on the straight road. But on the curves, it still drive directly to the water. Then I tried to get more data, three cameras, flipped, drive many laps. But the car still can't pass the first curve. The I collect data on the curce again and again, the size of data reached 20k, it finally passed the first two curves. The I collect data on the third curve repeadly, untill it pass it. Repead these steps, the car finally can drive smoothly around the whole track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
The mse always small on both data set, but it doesn't stop the car drving out of the drack. So I think the model wasn't underfitting or overfitting, just the training data is not good enough, or just not enough. So I get more and more training data, especially where it runs out of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is described before.



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The the images for left and right cameras are used, these will help keeping car in the center of the road.
![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back from the side of the road. These images show what a recovery looks like starting from left side and right side of the road:


![alt text][image5]
![image6]


To augment the data set, I also flipped images and angles thinking that this would increase the total number of training data. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]



After the collection process, I had 31376 data points. I then preprocessed this data by normalization: pixel / 255.0 - 0.5.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 because the mse does not change too much after 3 epochs, and less epochs can save a lot of time. I used an adam optimizer so that manually training the learning rate wasn't necessary.
