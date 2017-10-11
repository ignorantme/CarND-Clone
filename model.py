import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense,Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D

lines = []
with open('capture4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            path = 'capture4/IMG/'
            correction = 0.12

            for line in batch_samples:
                img_center_path = path + line[0].split('/')[-1]
                img_left_path = path + line[1].split('/')[-1]
                img_right_path = path + line[2].split('/')[-1]
                img_center = cv2.imread(img_center_path)
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                img_left = cv2.imread(img_left_path)
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                img_right = cv2.imread(img_right_path)
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                measurements.append(steering_center)
                measurements.append(steering_left)
                measurements.append(steering_right)

                image_flipped = np.fliplr(img_center)
                steering_flipped = -steering_center
                images.append(image_flipped)
                measurements.append(steering_flipped)

#                image_flipped = np.fliplr(img_left)
#                steering_flipped = -steering_left
#                images.append(image_flipped)
#                measurements.append(steering_flipped)

#                image_flipped = np.fliplr(img_right)
#                steering_flipped = -steering_right
#                images.append(image_flipped)
#                measurements.append(steering_flipped)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)





def LeNet():
    def resize_images(img):
        import tensorflow as tf
        return tf.image.resize_images(img,(32,32))
    model = Sequential()
    #model.add(Lambda(resize_images, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160,320,3)))

    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(6, 5, 5, name='conv1', subsample=(1, 1), border_mode="valid",activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(16, 5, 5, name='conv2', subsample=(1, 1), border_mode="valid", activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120, name='hidden1', activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(84, name='hidden2',activation="elu"))
    model.add(Dense(1, name='steering_angle'))
    return model


def nVidia():
    def resize_images(img):
        import tensorflow as tf
        return tf.image.resize_images(img,(160,320))

    model = Sequential()
#   model.add(Lambda(resize_images, input_shape=(160,320,3)))

    model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))
#    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Convolution2D(24,5,5, border_mode="same",subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, border_mode="same",subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, border_mode="valid",subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, border_mode="valid",activation='relu'))
    model.add(Convolution2D(64,3,3, border_mode="valid",activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# Preprocess incoming data, centered around zero with small standard deviation
#model = LeNet()
model = nVidia()
model.compile(loss='mse', optimizer= 'adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)



model.save('model.h5')
