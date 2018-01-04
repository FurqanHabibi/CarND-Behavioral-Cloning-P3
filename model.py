import csv
import cv2
import numpy as np
from tqdm import tqdm
import pickle

## Parameters
print('set parameters...')
side_cam_correction = 0.2
top_crop = 50
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

# read the image files
print('read image files...')
X_train = np.zeros((len(data), 160, 320,3))
y_train = np.zeros((len(data)))

for item, index in enumerate(tqdm(data)):
    image = cv2.cvtColor(cv2.imread(item[0]), cv2.COLOR_BGR2RGB)
    if (item[2] == 'flip'):
        image = np.fliplr(image)
    X_train[index] = image
    y_train[index] = float(item[1])

## Create the model
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense

print('creating model...')
# use the simpler sequential model
model = Sequential()
# crop the input image
model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0,0)), input_shape=(160, 320, 3)))
# normalize the image so it ranges from -1 to 1
model.add(Lambda(lambda x: ((x / 255.0) - 0.5) * 2))
# use a simple fully connected layer
model.add(Flatten())
model.add(Dense(1))
# use adam for optimizer and mean-squared-error for loss
model.compile(optimizer='adam', loss='mse')
# train the model
print('training...')
model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)
# save the model
model.save('model.h5')
