from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model


from keras.layers import Input, Dense
from keras.models import Model


#import matplotlib.pyplot as plt

batch_size = 8
num_classes = 4
epochs = 50
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
#inp = BatchNormalization()(inp)

##### Autoencoder starts #####
encoding_dim = 256
origin_dim = 137502
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(inp)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(origin_dim, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=inp, output=decoded)
# this model maps an input to its encoded representation
encoder = Model(input=inp, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train /= np.max(x_train)
x_test /= np.max(x_test)

# fit our autoencoder!
autoencoder.fit(x_train, x_train,
	nb_epoch=50,
	batch_size=256,
	shuffle=True,
	validation_data=(x_test, x_test))	
##### Autoencoder ends	#####
	
tr_encoded_imgs = encoder.predict(x_train)
print(tr_encoded_imgs.shape) 	
	
te_encoded_imgs = encoder.predict(x_test)
print(te_encoded_imgs.shape)		



inp2 = keras.layers.Input(shape=(encoding_dim,))


x = BatchNormalization()(inp2)

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

model = keras.models.Model(inputs=inp2, outputs=y)

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



history = model.fit(tr_encoded_imgs, y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(te_encoded_imgs, y_test), #    validation_split=.3, #
      shuffle=True)
	  
	  
print(history.history.keys())

import pickle
	
with open('autoencoder_timeseries.log', 'w') as file_pi:
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
