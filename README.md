# Project: Build a Traffic Sign Recognition Program

# Introduction:

In this project, Convoluted and Deep neural Networks were implemented using Python Library Tensorflow. 
This is part of the Term-1 Project-2 of Udacity Nanodegree program, where the concepts of Deep Neural Networks, Convolutional Neural Networks are dealt. The sub-sections explain various steps involved and taken in this project.

## Data Exploration:
The files were already given in pickle format and the pickle files are loaded, and filters were applied using pickle package for Pythonn.
The data is already divided into 3 different categories, namely Test, Training and Validation. Training data is used for training the network, test and validation for testing and validation respectively.
Visualization os the data is done by plotting various images present in the data using functions such as matplotlib. subplots were used and the necessary plots were generated.

Data Statistics: Number of Training Examples: 34799
                 Number of Testing Examples: 12630
                 Image Data Shape in DataSet: (32,32,3)
                 Number of Classes: 43
                 
Sample Images of the DataSet are given below:

<a href="url"><img src="https://image.ibb.co/jiRYWk/before_persp.png" align="center" height="200" width="250" ></a>

## Design and Test Model Architecture:
Preprocessing: For this step, simple normalization of the data is done. i.e., simply each of the image file is converted to zero mean and 1 variance by subtracting 128 and dividing the remainder by 128.
Model Architecture: Convoluted Neural Network very similar to Lenet is implemented. In the first step, The input file in format 32*32*3 is converted to 28*28*16. This is done by kernel of (5,5). Then max-pooling with strides of (2,2) is done. After this, again 2D convolution of kernel 5*5 is done to obtain image size of 10*10*64
Then, again max-pooling is applied for the image. Obtained size is 5*5*64.

After max-pooling, flattening of the image is done using flatten function. The obtained size is 1600, fully connected.

Then, dropout followed by further truncation of size to 120 is done.

After this step, dropout followed by truncation to 84, followed by dropout again and final tuncation to 43 (no. of classes) is done.

Note that after each step, relu activation is implemented. Also, finally softmax activation is implemented for web images.

## Model Training:
For training of the model, 20 epochs were used and Adam optimizer is used for training data. The optimizer is given the loss function as input which is calculated from cross entropy of training data and one hot labels of the output expected. Accuracy of about .99 for both training and testing data, followed by 0.971 for validation data is observed.

## Solution:
As explained earlier, the solution obtained was much higher than 0.93. Its about 0.99 for both training and testing and about 0.97 for validation.

# Test Model Against new images:

## Acquiring new Images:
New images were acquired using internet search for German Images in Google and downloaded and saved in a folder from where it can be accessed.

## Performance against new Images:
The trained model is saved using the tensorflow save session function. Then, saved model is restored and the parameters were tested against the new images. The performance of the images was okay and about 3/5 images were correctly identified.

## Model Certainty:
The models which were identified correctly were identified with very high confidence level of probability over 98% when used softmax activation (exponential). This suggests that model is trained well for image classification.

# Results:

The accuracy of the training set is # 99%, accuracy of test set is also # 99% and validation set is having an accuracy of about # 97.1%. 
The details of implementation can be seen in Ipython Notebook attached.

# Observations

Though the accuracy of the data is over 97% of the data provided, for new set of images which were downloaded from internet, the accuracy is about 60%, which is not to the expectations. This is mainly because the neural net is tuned to the variables which are present in the data and is not able to process new images accuractely. One way to handle this issue is by training the data over larger variety of data and also analyzing what are present in each layer of the image and then selecting images which cover all possible configurations
