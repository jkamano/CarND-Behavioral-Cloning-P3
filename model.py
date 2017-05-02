import numpy as np
import csv
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_data_paths(datafolder = '../data'):
    '''
       Load all metadata from csv file.

       Returns list with csv lines for all dataset given inside datafolder.
       The datafolder structure should be:
           <datafolder>/<DataSet1>/IMG/...
           <datafolder>/<DataSet1>/driving_log.csv
           <datafolder>/<DataSet2>/IMG/...
           <datafolder>/<DataSet2>/driving_log.csv
           ...
       DataSet# is the folder as exported by the simulator, multiple can be given as training/validation/test set

    '''
    # Load dataset
    # List data folders
    folders = next(os.walk(datafolder))[1]

    # Save all csv lines with relative paths
    csvlist = []
    for f in folders:
        with open(os.path.join(datafolder, f, "driving_log.csv")) as csv_fid:
            contents = csv.reader(csv_fid)
            for l in contents:
                for n in range(3):
                    l[n] = os.path.join(datafolder, f, "IMG", l[n].split('/')[-1])
                csvlist.append(l)

    return csvlist

def load_data(csvlist, addFlipped=True, addSideCameras=True):
    '''
        Load data. Full dataset loading
    '''
    X_train = []
    y_train = []
    for l in csvlist:
        img = cv2.imread(l[0])
        X_train.append( img )
        y_train.append(float(l[3]))

        if addFlipped:
            X_train.append(np.fliplr(img))
            y_train.append(-float(l[3]))

        if addSideCameras:
            offset = 0.15
            X_train.append(cv2.imread(l[1]))
            y_train.append(float(l[3]) + offset)
            X_train.append(cv2.imread(l[2]))
            y_train.append(float(l[3]) - offset)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return (X_train, y_train)

def generator(samples, batch_size=32, addFlipped=True, addSideCameras=True):
    '''
    yields subsets of training/validation dataset instead of full dataset to reduce RAM usage
    '''

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img = cv2.imread(batch_sample[0])
                images.append(img)
                angles.append(float(batch_sample[3]))
                if addFlipped:
                    images.append(np.fliplr(img))
                    angles.append(-float(batch_sample[3]))
                if addSideCameras:
                    offset = 0.15
                    images.append(cv2.imread(batch_sample[1]))
                    angles.append(float(batch_sample[3]) + offset)
                    images.append(cv2.imread(batch_sample[2]))
                    angles.append(float(batch_sample[3]) - offset)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

def define_model(in_shape = (160,320,3)):
    '''
        Defines model and optimizer.
    '''

    #model = testNet(in_shape = (160,320,3))
    #model = LeNet(in_shape = (160,320,3))
    model = nvidia(in_shape = (160,320,3))

    #setup optimizer and loss function to be used
    model.compile(loss="mse", optimizer="adam")

    return model

def testNet(in_shape = (160,320,3)):
    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=in_shape))
    model.add(Flatten())
    model.add(Dense(1))

    return model

def LeNet(in_shape = (160,320,3)):
    '''
    Create LeNet architecture.
    '''

    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=in_shape)) #normalize and center
    model.add(Cropping2D( cropping=((65, 25), (0, 0)) ))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(120))
    model.add(Dropout(0.4))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def nvidia(in_shape = (160,320,3)):
    '''
        Replicate nvidia network with keras
    '''
    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=in_shape)) #normalize and center
    model.add(Cropping2D( cropping=((65, 25), (0, 0)) ))
    model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5, activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5, activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3, activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3, activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def train(model, csvlist, epochs=10, modelfilename="model.h5"):
    '''
        Train model with dataset passed.
    '''
    addFlipped = True
    addSideCameras = True
    useGenerators = True
    valSplit = 0.2

    if useGenerators:
        train_samples, validation_samples = train_test_split(csvlist, test_size=valSplit)
        # compile and train the model using the generator function
        train_generator = generator(train_samples, batch_size=32, addFlipped=addFlipped, addSideCameras=addSideCameras)
        validation_generator = generator(validation_samples, batch_size=32, addFlipped=addFlipped, addSideCameras=addSideCameras)
        factor = 1
        if addFlipped:
            factor += 1
        if addSideCameras:
            factor += 2
        history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*factor, validation_data=validation_generator, \
                nb_val_samples=len(validation_samples)*factor, nb_epoch=epochs)
    else:
        (X_train, y_train) = load_data(csvlist, addFlipped=addFlipped, addSideCameras=addSideCameras)
        history = model.fit(X_train, y_train, validation_split=valSplit, shuffle=True, nb_epoch=epochs)

    model.save(modelfilename)

    return history



if __name__ == "__main__":

    print('Loading...')
    csvlist = load_data_paths('../data')

    model = define_model()

    print('Training...')
    history = train(model, csvlist, epochs=7)

    #From Jason Brownlee http://machinelearningmastery.com
    # summarize history for loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.svg',bbox_inches="tight" )
    plt.savefig('loss.png',bbox_inches="tight" )
