import cv2
import csv
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def findImages(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        lines = getLinesFromDrivingLogs(directory, skipHeader=True)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
            
def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu", W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu", W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu", W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation="relu", W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation="relu", W_regularizer=l2(0.001)))
    # model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    return model

if __name__ == '__main__':
    # Reading images locations.
    centerPaths, leftPaths, rightPaths, measurements = findImages('./data')
    imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
    print('Total Images: {}'.format( len(imagePaths)))

    # Splitting samples and creating generators.
    samples = list(zip(imagePaths, measurements))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    train_generator = generator(train_samples, batch_size=16)
    validation_generator = generator(validation_samples, batch_size=16)

    # Model creation
    model = nVidiaModel()
    print(model.summary())

    # Compiling and training the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
                                        validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                        nb_epoch=3, verbose=1)

    model.save('model.h5')

    # Save model architecture as json file
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

