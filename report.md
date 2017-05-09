#**Behavioral Cloning**

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/T1.png "Track1"
[image3]: ./examples/T1f.png "Track1"
[image4]: ./examples/T2_1.png "Track2"
[image5]: ./examples/T2_1d.png "Darker version"
[image6]: ./examples/T2_2.png "Track2"
[image7]: ./examples/T2_2l.png "Lighter version"
[image8]: ./examples/loss.png "Loss_vs_epoch1"
[image9]: ./examples/loss_final.png "Loss_vs_epoch3"

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started testing the LeNet5 model and then decided to do a replication of NVIDIA model described in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/, it is defined in _nvidia()_ (line 213) and represented in the following picture:
![alt text][image1]


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 227, 229, 231).

Exaustive testing was done to make sure the model was able to drive around the track.
Initially it was overfitting but by limiting the number of epochs to three the model is now driving well around the 1st track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 182).
The probability of dropping was set to 0.5 adopting the values from last assignment for the traffic sign recognition. Apparently it works.

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

* Added recovery situations. Still applying to LeNet5.
Still, LeNet5 network seems to be insuficient to model all possible situations and adapt to it. I moved to setup a new, more complex, model.

#####2) NVIDIA network
* Initialy nvidia model didn't perform too well. It had problems on the curve with dirt road, it would not see the dirt as a limit of the street and just continue straight.
I included a few more recordings of dirt curves and augmented the dataset by including images with random light intensity.

* I also recorded laps from the second track, hopping this would help generalize the model to improve on track 1 but also eventually to drive track 2.
Unfortunatelly the model I deliver  is only capable of driving track1.
Track to is really dificult even for me to drive. It also has the challenge of speed control which is not quite managed by the PI control. In certain descends the car breaks and blocks, and altough I see my velocity setpoint increasing, the car never moves again. (maybe some bug on the simulator?)

Nevertheless, all models I've achieved capable of driving on the 2nd track performed very poorly on track 1.
To drive in track 2 I actually removed every example of the dataset with


####2. Final Model Architecture

The final model is the one used by nvidia. I managed to train it to drive track1 and track 2 fairly well (reached the bridge) unfortunatelly never both. Either the model performed well on track 1 or track 2. Never both. Is this an overfitting problem perhaps?

####3. Creation of the Training Set & Training Process
* I recorded initialy 2 laps on track 1, one in normal sense and other in opposite sense.
* Has most of the track is straight I recorded extra situations only in curves or recovering from line to the center of the lane. Also some specially on the dirt curve which seemed a problem from the model to detect.

* Recorded track 2, (difficult without crashing), one lap normal sense and other opposite sense. Always in the left lane.
As this track is so full of curves, didn't do any special "recovery" recording.

* Augmented the data by adding random light intensity, adding left and right cameras and adding horizontally flipped images of the first track. (Not second in order to drive always on left lane).

This gave a total of 52485 features, randomly suffled and divided, 80% taken for train and 20% for validation.

These are example images used:

* 1st row is from 1st track, acquired and flipped
* 2nd row is 2nd track, acquired and preprocessed darker version
* 3rd row is 2nd track, acquired and preprocessed lighter version

![alt text][image2]  ![alt text][image3]
![alt text][image4]  ![alt text][image5]
![alt text][image6]  ![alt text][image7]


The ideal number of epochs was 3 as evidenced by following pictures that show the loss over the epochs for training and testing set on two training experiments.
This was also confirmed by saving the models on each epoch and testing the each on the track.

Second picture corresponds to the actual training of the model delivered.
![alt text][image8]
![alt text][image9]
