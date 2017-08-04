import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils

np.random.seed(1671)  # for reproducibility

# Network and training
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # number of outputs = number of digits
N_HIDDEN = 1024
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# Data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors  to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10 classes
# final stage is softmax
# NB_EPOCH = 200
# OPTIMIZER = SGD()  # SGD optimizer
# model = Sequential()
# model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
# model.add(Activation('softmax'))
# model.summary()

NB_EPOCH = 25
OPTIMIZER = Adam()  # SGD optimizer
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# Model compiles
model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy'])

# Model training
history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print()
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
