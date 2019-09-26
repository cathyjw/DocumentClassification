import os,cv2
from keras import backend
from keras import callbacks
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
backend.set_session(sess)

trainDatagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

testDatagen = ImageDataGenerator(rescale=1./255)
trainDataset = trainDatagen.flow_from_directory(
        'data/train',
        target_size=(299,299),
        batch_size=16,
        class_mode='binary')

testDataset = testDatagen.flow_from_directory(
        'data/test',
        target_size=(299,299),
        batch_size=2,
        class_mode='binary')
num_classes=2      

# CNN Model
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(299,299,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration
print(model.summary())

# Fit the model
history=model.fit(trainDataset, steps_per_epoch=335, epochs=4, verbose=1, validation_steps=4, validation_data=testDataset)

test_image=cv2.imread('data/P76804167HAB_PMT_EFF_20190808_01_BRO.jpg')
test_image = test_image.reshape(1,299,299,3)
pred=model.predict(test_image, steps=1)
print("Probability that it is a PMT = ", "%.2f" % (1-pred))

model_json=model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
