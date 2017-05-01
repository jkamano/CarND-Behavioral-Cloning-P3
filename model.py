import numpy as np
import csv
import os
import cv2
import matplotlib.pyplot as plt

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
        Load data. Using Generator?
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def define_model(in_shape = (160,320,3)):
    '''
        Defines model and optimizer.
    '''

    model = LeNet(in_shape = (160,320,3))

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
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def train(model, X_train, y_train, epochs=10, modelfilename="model.h5"):
    '''
        Train model with dataset passed. Validation split of 20% and shuffle
    '''

    history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

    model.save(modelfilename)

    return history

if __name__ == "__main__":

    print('Loading...')
    csvlist = load_data_paths('../data')
    (X_train, y_train) = load_data(csvlist, addFlipped=True, addSideCameras=True)



    model = define_model()

    print('Training...')
    history = train(model, X_train, y_train, epochs=3)

    #From Jason Brownlee http://machinelearningmastery.com
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
