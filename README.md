# Project: Build a Traffic Sign Recognition Program

Introduction:

In this project, Convoluted and Deep neural Networks were implemented using Python Library Tensorflow. The brief details followed to complete this project were given below:

The Project
The goals / steps of this project are the following:
* Load the data set
For reading csv files, pandas package is used. Reading and printing images is done through matplotlib imread and imshow. Additionally the pickle files were loaded using pickle package of python
* Explore, summarize and visualize the data set
This is done mainly using Matplotlib
* Design, train and test a model architecture
Tensorflow lenet architecture is used for this part of the project. The solution of Lenet given coupled with dropout and max pooling is done.
* Use the model to make predictions on new images
New images were downloaded from internet and were analyzed for results.
* Analyze the softmax probabilities of the new images
This is done using bar function of matplotlib.


### Results:

The accuracy of the training set is # 99%, accuracy of test set is also # 99% and validation set is having an accuracy of about # 97.1%. 
The details of implementation can be seen in Ipython Notebook attached.

### Observations

Though the accuracy of the data is over 97% of the data provided, for new set of images which were downloaded from internet, the accuracy is about 60%, which is not to the expectations. This is mainly because the neural net is tuned to the variables which are present in the data and is not able to process new images accuractely. One way to handle this issue is by training the data over larger variety of data and also analyzing what are present in each layer of the image and then selecting images which cover all possible configurations
