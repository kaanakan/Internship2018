from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import pickle
#import matplotlib.pyplot as plt

batch_size = 8
num_classes = 4
epochs = 500
#epochs = 5000
#epochs = 8000
#epochs = 10000
data_augmentation = False

# The data, shuffled and split between train and test sets:
import numpy as np
import scipy.io

#x_test = scipy.io.loadmat('x_test.mat')['x_test']
#x_train = scipy.io.loadmat('x_train.mat')['x_train']
#x_test = scipy.io.loadmat('teData_mean.mat')['teData_mean']
#x_train = scipy.io.loadmat('trData_mean.mat')['trData_mean']
x_test = scipy.io.loadmat('teData_raw.mat')['teData_raw']
x_train = scipy.io.loadmat('trData_raw.mat')['trData_raw']
y_test = scipy.io.loadmat('y_test.mat')['y_test']
y_train = scipy.io.loadmat('y_train.mat')['y_train']

y_train[y_train == 1] = 0
y_train[y_train == 3] = 1
y_train[y_train == 5] = 2
y_train[y_train == 7] = 3

y_test[y_test == 1] = 0
y_test[y_test == 3] = 1
y_test[y_test == 5] = 2
y_test[y_test == 7] = 3

#x_train = x_train.reshape(210,-1)
#x_test = x_test.reshape(210,-1)

y_train = y_train.reshape(210)
y_test  = y_test .reshape(210)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#inp = keras.layers.Input(shape=(47*47*47,))
#inp = keras.layers.Input(shape=(22917,))
inp = keras.layers.Input(shape=(137502,))
x = BatchNormalization()(inp)

x = Dense(32, kernel_regularizer=keras.regularizers.l2(0.01))(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Activation('tanh')(x)
#x = Activation('sigmoid')(x)

x = Dense(16, kernel_regularizer=keras.regularizers.l2(0.01))(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Activation('tanh')(x)
#x = Activation('sigmoid')(x)

#x = Dense(8, kernel_regularizer=keras.regularizers.l2(0.01))(x)
##x = BatchNormalization()(x)
#x = Activation('relu')(x)

x = Dense(num_classes)(x)
y = Activation('softmax')(x)

model = keras.models.Model(inputs=inp, outputs=y)

#model.add(Dropout(.4))

#model.add(Dense(num_classes))
#model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
opt = keras.optimizers.Adam(lr=.000001)
#opt = keras.optimizers.SGD(lr=.0000001,momentum=0.9,nesterov=True)

#model = load_model('my_model.h5')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

history = model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_test, y_test), #    validation_split=.3, #
      shuffle=True)
	  
	  
print(history.history.keys())

import pickle
	
with open('trainlog_timeseries.log', 'w') as file_pi:
	pickle.dump(history.history, file_pi)	  

# model.save('my_model.h5')

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
