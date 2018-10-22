from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization,GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

from keras.layers import Input, Dense
from keras.models import Model

# import matplotlib.pyplot as plt

batch_size = 21
num_classes = 4
epochs = 6000
data_augmentation = False

# The data, shuffled and split between train and test sets:
import numpy as np
import scipy.io

# x_test = scipy.io.loadmat('x_test.mat')['x_test']
# x_train = scipy.io.loadmat('x_train.mat')['x_train']
# x_test = scipy.io.loadmat('teData_mean.mat')['teData_mean']
# x_train = scipy.io.loadmat('trData_mean.mat')['trData_mean']
x_test = scipy.io.loadmat('teData_raw_cnn.mat')['teData_raw_cnn']
x_train = scipy.io.loadmat('trData_raw_cnn.mat')['trData_raw_cnn']
y_test = scipy.io.loadmat('tr_te_labels_4class.mat')['te_labels_four_class']
y_train = scipy.io.loadmat('tr_te_labels_4class.mat')['tr_labels_four_class']

y_train[y_train == 1] = 0
y_train[y_train == 3] = 1
y_train[y_train == 5] = 2
y_train[y_train == 7] = 3

y_test[y_test == 1] = 0
y_test[y_test == 3] = 1
y_test[y_test == 5] = 2
y_test[y_test == 7] = 3

# x_train = x_train.reshape(210,-1)
# x_test = x_test.reshape(210,-1)

y_train = y_train.reshape(210)
y_test = y_test.reshape(210)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# inp = keras.layers.Input(shape=(47*47*47,))
# inp = keras.layers.Input(shape=(22917,))
# inp = keras.layers.Input(shape=(137502,))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

inp2 = keras.layers.Input(shape=(48,48,48,6))
x = BatchNormalization()(inp2)

x = GaussianNoise(0.01)(x)


x = Conv3D(32,5,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)



x = MaxPooling3D(2)(x)

x = Conv3D(32,3,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)


x = MaxPooling3D(2)(x)


x = Flatten()(x)

x = Dense(1024)(x)

x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# x = Activation('tanh')(x)
# x = Activation('sigmoid')(x)

x = Dense(256)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)

x = Dropout(0.2)(x)

# x = Activation('tanh')(x)
# x = Activation('sigmoid')(x)

x = Dense(64)(x)
x = Activation('relu')(x)

x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(num_classes)(x)
y = Activation('softmax')(x)

model = keras.models.Model(inputs=inp2, outputs=y)


opt = keras.optimizers.Adam(lr=.00001)
# opt = keras.optimizers.SGD(lr=.0000001,momentum=0.9,nesterov=True)

# model = load_model('my_model.h5')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=20,
                    validation_data=(x_test, y_test),  # validation_split=.3, #
                    shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())

import pickle

with open('s7_conv_enc.log', 'w') as file_pi:
    pickle.dump(history.history, file_pi)
