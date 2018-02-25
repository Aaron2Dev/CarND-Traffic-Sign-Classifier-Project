# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[BarChart]: ./examples/BarChart.JPG "BarChart"
[Dataset_imgs]: ./examples/Dataset_exploration.JPG "Dataset examples"
[Grayscale]: ./examples/Grayscale.JPG "Grayscale"
[Augmented]: ./examples/Augmented.JPG "Augmented"
[new_imgs]: ./examples/new_imgs.JPG "new_imgs"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of initial training set is 34799.
* The size of the initial validation 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. You can see 10 random images out of the test set.
The Barchart shows the frequency of occurence of the Traffic signs.

![alt text][Dataset_imgs]
![alt text][BarChart]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. In the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" from Pierre Sermanet and Yann LeCun the CNN performed better by grayscaled images than on color images.

Here are some examples before and after grayscaling:

![alt text][Grayscale]

As a last step, I normalized the image data. (It was suggested and is usually a basic preprocessing operation on images.)

Also I generated additional data. I tried first to train the CNN without any data augmentation and the performance. Since the performance was not high enough, I decided to augment the data set.

To generate more data, I used basic techniques like translation, rotation, scaling and changing the brightness. In order to this I used openCV.
Every Traffic sign with under 800 samples gets augmented.

Here is an example of some original images and and some augmented images:

![alt text][Augmented]

The Augmentation process included the following:
* Translation in a range of -3,+3 pixels
* Rotation in a range of -30,+30 degrees
* Scaling - I used cv2.getPerspectiveTransform() and cv2.warpPerspective()
* Brightness - here I the basically tried some different alpha and beta values and ended up with an alpha between [1-1.5] and a beta [-50,50]

After the augmentation process the size of my training set was 46480.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet Architecture and just made a few changes.


* Input
* Convolution Layer1
* RELU
* Max pooling
* Convolution Layer2
* RELU
* Max pooling
* Flatten
* Fully Connected Layer 3
* RELU
* (Dropout)					
* Fully Connected Layer 4
* RELU
* (Dropout)
* Fully Connected Layer 5


The Input layer consisted of grayscale images with shape (32,32,1).
For the activation function I tried RELU as well as tanh(). After some tries and reading I decided to use the RELU function.
I also added to dropout Layer to the architecture. Only after the convolutinal part.


#### 3.Hyperparameters

* Epochs: 15
* The optimizer, learning rate and batch size I leaved unchanged.
* For the dropout I used and a keeping probability of 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97% 
* test set accuracy of 89,3%

I started my approach with the basic LeNet architecture. Based on the accuracy improvement I tried different methods, which were suggested in the Traffic sign Classification Paper.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new_imgs]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)  | Speed limit (50km/h)  						| 
| Road work    			| Road work  									|
| No entry				| No entry										|
| No passing	      	| No passing					 				|
| Priority road			| Priority road	      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The only one which was missclassified is the 60km/h speed limit and it was classified as a 50km/h speed limit sign.(They look quite similiar)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|89,7%					|Speed limit (50km/h							|
|100%       			| Road work  									| 
|99.6%      			| No entry 										|
|75%					| No passing									|
|99.9%	      			| Priority road					 				|



