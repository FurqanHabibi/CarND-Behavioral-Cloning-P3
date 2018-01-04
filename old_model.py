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

## Load data

# read csv files
print('read csv files...')
data_folders = ['data/']

lines = []
for folder in data_folders:
    with open(folder + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            line.append(folder)
            lines.append(line)
            
# read images and steering angles
print('read images and steering angles...')
images = []
steering_angles = []
for line in tqdm(lines):
    img_center = cv2.cvtColor(cv2.imread(line[-1] + line[0].strip()), cv2.COLOR_BGR2RGB)
    print(img_center.shape)
    img_left = cv2.cvtColor(cv2.imread(line[-1] + line[1].strip()), cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(line[-1] + line[2].strip()), cv2.COLOR_BGR2RGB)

    steering_center = float(line[3])
    steering_left = steering_center + side_cam_correction
    steering_right = steering_center - side_cam_correction

    images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])

## Data Augmentations
images = np.array(images)
steering_angles = np.array(steering_angles)

X_train = np.zeros((24108 * 2, 160, 320,3))
y_train = np.zeros((len(images) * 2))

# add flipped images
print('flip images...')
y_train[0:len(images):] = steering_angles
y_train[len(images)::] = steering_angles * -1
X_train[0:len(images):] = images
X_train[len(images)::] = np.fliplr(images)

# save to pickle file
with open('data.pickle', 'wb') as pickle_file:
    pickle.dump((X_train, y_train), pickle_file)

# load from pickle file
X_train, y_train = None, None
with open('data.pickle', 'rb') as pickle_file:
    (X_train, y_train) = pickle.load(pickle_file)

## Create the model
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense

print('creating model...')
# use the simpler sequential model
model = Sequential()
# crop the input image
model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0,0)), input_shape=(3,160,320)))
# normalize the image so it ranges from -1 to 1
model.add(Lambda(lambda x: ((x / 255.0) - 0.5) * 2))
# use a simple fully connected layer
model.add(Flatten())
model.add(Dense(1))
# use adam for optimizer and mean-squared-error for loss
model.compile(optimizer='adam', loss='mse')
# train the model with 20% validation split and shuffling
# use the default 32 batch size and 10 epoch
print('training...')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
# save the model
model.save('model.h5')
