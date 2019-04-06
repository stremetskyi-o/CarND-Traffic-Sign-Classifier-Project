# **Traffic Sign Recognition** 
## Project Writeup

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

[dataset-example]: ./project-writeup-img/dataset_example.png "Dataset example"
[augmented-example]: ./project-writeup-img/augmented_example.png "Augmented example"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[test1]: ./test_images/1_13.jpg "Traffic Sign 1"
[test2]: ./test_images/2_4.jpg "Traffic Sign 2"
[test3]: ./test_images/3_2.jpg "Traffic Sign 3"
[test4]: ./test_images/4_12.jpg "Traffic Sign 4"
[test5]: ./test_images/5_17.jpg "Traffic Sign 5"

---

### 1. Writeup

This writeup describes research and implementation steps that were taken to address project rubric points. Where 
appropriate steps are backed up by additional information in the form of images, tables, etc.

### 2. Dataset Exploration

#### 2.1. Dataset Summary

Dataset contains 4 different properties and is split to training, validation and testing subsets. The 4 properties are:
*features*, *labels*, *sizes* and *coords*. The first two contains resized pictures of the road signs and their corresponding
labels. Latter two, information about raw images.

I have used capabilities of the NumPy library to receive a summary of the dataset features and labels:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

#### 2.2. Exploratory Visualization

I have considered useful to see at least one example image of each class to have understanding what kind of features
classifier should be looking at. For this NumPy `unique` method was used on dataset labels with `return_index=True`
so it is possible to map values to indexes of the images. The following figure is the plot of single sign image for each
class.

![dataset-example]

From the figure it seen that signs have different shape, color and inner image or text, in addition some signs are crossed.
In details there 4 different colors, 5 shapes, 38 numbers/drawings and 2 states for crossed lines. In general images have different background and lighting conditions.

### 3. Design and Test a Model Architecture

#### 3.1. Preprocessing

I have left images RGB, because at some basic level signs can be easily split to categories by the color alone; in addition, because of 
different lighting conditions and small image size grayscaling may remove some detail.

Generating augmented data is common technique used to improve the quality of the classifier when the dataset is relatively small.
Original dataset already contain images under different lighting conditions, so I have decided to improve it by generating additional images with a randomly changed perspective.
The sign can be observed at different angles from left to right but typically at some common height, because vehicles don't vary much by height and signs are mounted at specific range of heights.  

To implement this augmentation I have used the data in `sizes` and `coords` to convert original coordinates to resized and build source coordinates of the sign.
Than destination coordinates was calculated by squeezing left or right side of the sign in the range from 10% to 40% of the height. Transformation was performed using OpenCV.

An example of original and augmented image:

![augmented-example]
 
I have used only 1 additional image for each original one to prevent overfitting, which can be caused by having 3 or more visually similar images.
This operation doubled the size of the training dataset.

The next step I took is normalizing the input which dramatically improves the quality of the classifier by allowing to converge faster.
The formula I have used is `data / 128 - 1` this scales dataset values to the range *[-1; 1)*


#### 3.2. Model Architecture

LeNet-5 was used as the base of developed model with some modification to allow 3 channels input and 43 classes output.
To accommodate the need of the classifier size of the layers was increased.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![test1] ![test2] ![test3] 
![test4] ![test5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


