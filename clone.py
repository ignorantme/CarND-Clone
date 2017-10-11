import csv
import cv2
import numpy as np
import sklean
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D

lines = []
with open('capture2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    path = 'capture3/IMG/'   
    
    correction = 0.2

    img_center_path = path + line[0].split('/')[-1]
    img_left_path = path + line[1].split('/')[-1]
    img_right_path = path + line[2].split('/')[-1]
    img_center = cv2.imread(img_center_path)
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    
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

    image_flipped = np.fliplr(img_left)    
    steering_flipped = -steering_left
    images.append(image_flipped)
    measurements.append(steering_flipped)

    image_flipped = np.fliplr(img_right)    
    steering_flipped = -steering_right
    images.append(image_flipped)
    measurements.append(steering_flipped)
    

X_train = np.asarray(images)
y_train = np.asarray(measurements)


def resize_images(data, shape):
    resized = []
    for img in data:
        img = cv2.resize(img, shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized.append(img)
    return np.array(resized)
def generator(samples, batch_size=32):
    


def LeNet(input_shape=(32,32,3)):
    def resize_images(img):
        import tensorflow as tf
        return tf.image.resize_images(img,(32,32))
    model = Sequential()
    #model.add(Lambda(resize_images, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
 
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



model = LeNet()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')
