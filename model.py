import csv
import cv2
import numpy as np
from tqdm import tqdm
import pickle

## Parameters
print('set parameters...')
side_cam_correction = 0.2
top_crop = 60
bottom_crop = 20
validation_split = 0.2
batch_size = 32
nb_epoch = 10

## Load data

# read csv files
print('read csv files...')
data_folders = ['data/']

data = []
for folder in data_folders:
    with open(folder + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            # use all 3 cameras
            data.append([folder + line[0].strip(), float(line[3]), 'normal'])
            data.append([folder + line[1].strip(), float(line[3]) + side_cam_correction, 'normal'])
            data.append([folder + line[2].strip(), float(line[3]) - side_cam_correction, 'normal'])
            # use flipped image
            data.append([folder + line[0].strip(), -(float(line[3])), 'flip'])
            data.append([folder + line[1].strip(), -(float(line[3]) + side_cam_correction), 'flip'])
            data.append([folder + line[2].strip(), -(float(line[3]) - side_cam_correction), 'flip'])

# create a batch generator
print('creating generator...')
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle the data every epochs
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                if (batch_sample[2] == 'flip'):
                    image = np.fliplr(image)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

## Create the model
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Convolution2D, MaxPooling2D

print('creating model...')
# use the simpler sequential model
model = Sequential()
# crop the input image
model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0,0)), input_shape=(160, 320, 3)))
# normalize the image so it ranges from -1 to 1
model.add(Lambda(lambda x: ((x / 255.0) - 0.5) * 2))
# create the layers
model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same'))
model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same'))
#model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
#model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
#model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
#model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
#model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(256, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D())
model.add(Flatten())
print(model.layers[-1].output_shape)
#model.add(Dense(2048))
model.add(Dense(512))
model.add(Dense(1))
# use adam for optimizer and mean-squared-error for loss
model.compile(optimizer='adam', loss='mse')
# train the model
print('training...')
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data, test_size=validation_split)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)
# save the model
print('saving model...')
model.save('model.h5')
