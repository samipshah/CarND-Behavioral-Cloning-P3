**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/fig1.png "Input Data Histogram"
[image2]: ./writeup_images/fig2.png "Preprocessed Input Data"
[image3]: ./writeup_images/img.png "Normal Image"
[image4]: ./writeup_images/img_flip.png "Flipped Image"
[image5]: ./writeup_images/img_bright.png "Random Brightness Image"
[image6]: ./writeup_images/img_crop.png "Resized Image"
[image7]: ./writeup_images/img_yuv.png "YUV colorscheme"

## Rubric Points

---
## Required Files
#### 1. Submission includes Required Files
My submission has all the required files and a couple more 
* model.py contains the script to create and train the model
* read_data.py contains helper code for model.py to read training, validation data from disk using generator
* model.h5 contains the trained convolutional neural network
* writeup_report.md this file
* video.mp4 video of the trial run

## Quality of Code
#### 1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. read_data.py file has generator related code to read files in batches when required rather than reading all at once and keeping in memory. Code is well organized and commented

## Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed

I tried many different models, started with LeNet later on tried developing my own modifying Nvidia's model and then eventually ended up using NVidia's model architecture. Basically 5 convoluted layers, first 3 layers extracting 24, 36, 48 features respectively from 5x5 squares and with a stride of 2. Next 2 convoluted layers extracted 64 features from 3x3 squares with a stride of 1. Next 4 layers are fully connected layers reducing dimension to 10 and then output layer giving the correct steering angle. As Nvidia's architecture was proven went with that , some other architecture I had been trying would have worked as well, however this particular project has been so hair splitting, because correcting something using extra data did not really help that much. Right mix of data is what this project required. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used training data provided by Udacity. Because I tried collecting data numerous ways and then spending countless hours training it to be perfect, I later on resorted to Udacity's data. I'm not sure if keyboard data was particularly not good for training.

### Architecture and Training Documentation

#### 1. Solution Design Approach

I initially tried with the most basic Dense 1 layer network. Ofcourse it did not yield a good result. So later tried LeNet model which worked out to be smooth. However car was getting off road. So flipped images and measurement of already existing data, that helped train it slightly better. Later on figured out some ways of preprocessing data that could help train general model , applied random brightness, converted color scheme to YUV, flipped images randomly, used all three camera angles and applied a corrective factor of 0.25. Since there were a lot more straight road data than turns used mechanism that threw off 20% of straight data while training. Tried to get the training samples data histogram follow a flatter bell curve then a spike like curve. Preprocessing helped achieve right mix of data on which the model can generalize rather than overfit.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

1. Convolution layer extracting 24 features from 5x5 image with 2x2 stride.
2. Convolution layer extracting 36 features from 5x5 image with 2x2 stride.
3. Convolution layer extracting 48 features from 5x5 image with 2x2 stride 
4. Convolution layer extracting 64 features from 3x3 image
5. Convolution layer extracting 64 features from 3x3 image
6. Flatten layer
7. Fully connected layer with 1164 nodes
8. Dropout layer with 0.2 probability
9. Fully connected layer with 100 nodes
10. Fully connected layer with 50 nodes
11. Fully connected layer with 10 nodes
12. Output layer with 1 node

#### 3. Creation of the Training Set & Training Process

Before preprocessing input data histogram

![Input Data Histogram][image1]

After preprocessing input data histogram more uniformly distributed bell curve. Not heavily skewed towards remaining in straight line

![Preprocessed Input Data][image2]

#### Pre - Processing Steps

1. Normal image captured by the training is as below

![Normal Image][image3]

2. Randomly flips image 

![Image Flip][image4]

3. Random brightness applied to the image

![Image Brightness][image5]

4. Resized image to 66x200 as Nvidia architecture takes that
![Image Crop][image6]

5. Changed image to YUV colorscheme as it could be better for the model to train on
![Image YUV][image7]
