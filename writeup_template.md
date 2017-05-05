#**Behavioral Cloning**

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

#####1) LeNet5 network
* Used LeNet5 as explained in the classes.
* Hardly any response from the vehicle, it would just go straight.
* Augmented dataset as suggested, adding left and right camera and flipped images.
* Cropped images as suggested
* Hardly any response from the vehicle, it would just go straight.

* I found out that this is probably because most of the steering angles recorded with keyboard control are 0 degrees.
I moved to record my laps with mouse input which improved significantly the result.
The model was now able to drive to the first curve and some behavior was visible.

* To make the curves its visible that the model doesn't have enough curve/recovery examples in order to generalize. I moved to record new data just in recovery or curve situations.

* Added Drive4 and Drive5 folders. With recovery situations. Still applying to LeNet5.
_lenet5_1.h5_ is the trained LeNet model with Drive1,2,3 data directories and dropout of 40% on the fully connected layers. 10 epochs, looks good. Training looks good but it's not general enough. It fails some curves.

_lenet5_2.h5_ is the trained LeNet model with Drive1,2,3,4,5 data directories and dropout of 40% on the fully connected layers. 10 epochs, it overfits the data.


#####2) NVIDIA network
* _nvidia_1.h5_ is the trained nvidia model with Drive1,2,3,4,5 data directories and dropout of 40% on the fully connected layers. 10 epochs, it overfits the data.

* _nvidia_2.h5_ is the trained nvidia model with Drive1,2,3,4,5,6,7,8 data directories and dropout of 40% on the fully connected layers. 6 epochs. Drive6 introduces cases for the dirt road on track 1 and Drive 7 and 8 the second track.



####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.







Epoch 1/20
31285/31285 [==============================] - 647s - loss: 0.1471 - val_loss: 0.0913
Epoch 2/20
31285/31285 [==============================] - 778s - loss: 0.0900 - val_loss: 0.0902
Epoch 3/20
31285/31285 [==============================] - 771s - loss: 0.0867 - val_loss: 0.0838
Epoch 4/20
31285/31285 [==============================] - 772s - loss: 0.0819 - val_loss: 0.0809
Epoch 5/20
31285/31285 [==============================] - 660s - loss: 0.0789 - val_loss: 0.0767
Epoch 6/20
31285/31285 [==============================] - 717s - loss: 0.0763 - val_loss: 0.0753
Epoch 7/20
31285/31285 [==============================] - 659s - loss: 0.0746 - val_loss: 0.0728
Epoch 8/20
31285/31285 [==============================] - 643s - loss: 0.0726 - val_loss: 0.0707
Epoch 9/20
31285/31285 [==============================] - 687s - loss: 0.0700 - val_loss: 0.0695
Epoch 10/20
31285/31285 [==============================] - 638s - loss: 0.0680 - val_loss: 0.0699
Epoch 11/20
31285/31285 [==============================] - 637s - loss: 0.0665 - val_loss: 0.0682
Epoch 12/20
31285/31285 [==============================] - 637s - loss: 0.0657 - val_loss: 0.0676
Epoch 13/20
31285/31285 [==============================] - 641s - loss: 0.0640 - val_loss: 0.0659
Epoch 14/20
31285/31285 [==============================] - 645s - loss: 0.0610 - val_loss: 0.0660
Epoch 15/20
31285/31285 [==============================] - 645s - loss: 0.0598 - val_loss: 0.0643
Epoch 16/20
31285/31285 [==============================] - 647s - loss: 0.0575 - val_loss: 0.0630
Epoch 17/20
31285/31285 [==============================] - 661s - loss: 0.0556 - val_loss: 0.0626
Epoch 18/20
31285/31285 [==============================] - 659s - loss: 0.0554 - val_loss: 0.0630
Epoch 19/20
31285/31285 [==============================] - 659s - loss: 0.0544 - val_loss: 0.0668
