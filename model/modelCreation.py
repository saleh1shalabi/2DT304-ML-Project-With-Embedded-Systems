# Author: Saleh Shalabi
# Date: 2023-01-23


import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

#############################################################
# Defining the variables for the model #
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model
Conv1D = tf.keras.layers.Conv1D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
MaxPooling1D = tf.keras.layers.MaxPooling1D
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
#############################################################

path = '../data'


def convert_to_16k(file):
    """
    Converts the audio file to 16k sample rate.

    Args:
        file: the file path
    Returns:
        the audio file resampled in 16k
    """

    file_contents = tf.io.read_file(file)

    # decodes the file
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)

    # squeeze the audio file into a single dimension
    wav = tf.squeeze(wav, axis=-1)

    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    # resample the audio file to 16k from the original sample rate
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    return wav


def preprocess(file, label):
    """
    Preprocesses the audio file to a spectrogram.

    Args:
        file: the file path
        label: the label of the file
    Returns:
        the spectrogram of the audio file

    """
    wave = convert_to_16k(file)

    # the params as the frame length and frame step are chosen based on what could be assembled in the ESP32.
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=64,
                                 window_fn=tf.signal.hamming_window, pad_end=True)
    spectrogram = tf.abs(spectrogram)

    # the spectrogram is resized to 128x64 to match the size of the array in the ESP32
    # and because the audio was not crop to a specific length,
    # which means the spectrogram's from different audio files  will have different sizes
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.ensure_shape(spectrogram, [None, None, 1])
    spectrogram = tf.image.resize(spectrogram, [128, 64])
    spectrogram = tf.squeeze(spectrogram, axis=-1)

    # round the spectrogram to 2 decimal places.
    spectrogram = spectrogram * 100
    spectrogram = tf.round(spectrogram)
    spectrogram = tf.math.divide_no_nan(spectrogram, 100)

    # casting because the ESP32 doesn't have the same width and accuracy in numbers as python
    spectrogram = tf.cast(spectrogram, tf.float16)

    # the normalization was skipped due the computing limitations of the ESP32
    # spectrogram = spectrogram / tf.reduce_max(spectrogram)

    return spectrogram, label


#####################################################################################################
# get the path's for the audio files and create the dataset #
# the dataset is a list of tuples, each tuple contains the path to the audio file #
# and the label of the audio file #
# this will be made into a function later #
############################################################################

POS = os.path.join(path, 'positive')
NEG = os.path.join(path, 'NEG')

pos = tf.data.Dataset.list_files(POS + '/*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*.wav')

positive = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negative = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

data = positive.concatenate(negative)

######################################################################################################


######################################################################################################
# Test of plotting some spectrograms from the dataset #
# 10 plot will appear, each plot contains 2 spectrograms, #
# the first one is a positive spectrogram #
# the second one is a negative spectrogram #
######################################################################

# for i in range(10):
#     pos_file, pos_label = positive.shuffle(buffer_size=10000).as_numpy_iterator().next()
#     neg_file, neg_label = negative.shuffle(buffer_size=10000).as_numpy_iterator().next()

#     pos_specto, pos_label = preprocess(pos_file, pos_label)
#     neg_specto, neg_label = preprocess(neg_file, neg_label)

#     pos_specto = tf.transpose(pos_specto)
#     # pos_specto = tf.math.log(pos_specto)
#     neg_specto = tf.transpose(neg_specto)
#     # neg_specto = tf.math.log(neg_specto)

#     plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.imshow(pos_specto)
#     plt.subplot(2, 1, 2)
#     plt.imshow(neg_specto)

# plt.show()
######################################################################################################


######################################################################################################
# Preprocessing the dataset #
# and splitting it into train, validation and test sets #
# the train set is 80% of the dataset #
# the validation set is 20% of the train set #
# the test set is 20% of the dataset #
##############################################################

data = data.shuffle(len(data))
data = data.map(preprocess)
data = data.batch(32)
data = data.prefetch(32)

X = data.take(int(len(data) * 0.8))
test = data.skip(int(len(data) * 0.8)).take(int(len(data) * 0.2))

train = X.take(int(len(X) * 0.8))
val = X.skip(int(len(X) * 0.8)).take(int(len(X) * 0.2))

######################################################################################################


#######################################################################################################
# Creating the model #

model = Sequential()

model.add(Input(shape=(128, 64)))
model.add(Conv1D(4, 3, activation='relu', padding='same'))
model.add(Conv1D(8, 3, activation='relu', padding='same'))
model.add(Conv1D(16, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.001)))
model.add(Dense(1, activation='sigmoid'))

model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

print(model.summary())
print(len(data))

hist = model.fit(train,
                 epochs=100,
                 verbose=1,
                 validation_data=val)

model.save('../models/model.h5')
print("**************************Model saved**************************")

######################################################################################################


######################################################################################################
# Plotting the results #
# the results are the loss, precision, recall and accuracy #
###########################################################################


plt.title(f'Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.legend(['loss', 'val_loss'], loc='upper right')

plt.figure()
plt.title(f'Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.legend(['precision', 'val_precision'], loc='upper right')

plt.figure()
plt.title(f'Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.legend(['recall', 'val_recall'], loc='upper right')

plt.figure()
plt.title(f'Accuracy')
plt.plot(hist.history['binary_accuracy'], 'r')
plt.plot(hist.history['val_binary_accuracy'], 'b')
plt.legend(['binary_accuracy', 'val_binary_accuracy'], loc='upper right')

######################################################################################################
# Testing the model #
# the results are the loss, precision, recall and accuracy #
###########################################################################

right = 0
wrong = 0

right_ones = 0
wrong_ones = 0

right_zeros = 0
wrong_zeros = 0

# x = model.evaluate(test)
# print(x)

iter = test.as_numpy_iterator()

X_test, y_test = iter.next()

for i in iter:
    X_test = np.concatenate((X_test, i[0]), axis=0)
    y_test = np.concatenate((y_test, i[1]), axis=0)

yhat = model.predict(X_test)

yhat = [1 if i > 0.5 else 0 for i in yhat]

for i in range(len(yhat)):
    if yhat[i] == y_test[i]:
        if yhat[i] == 1:
            right_ones += 1
        else:
            right_zeros += 1

    if yhat[i] != y_test[i]:
        if yhat[i] == 1:
            wrong_ones += 1
        else:
            wrong_zeros += 1

    # print("predict:", yhat[i], " is: ", y_test[i])

print("right ones: ", right_ones, "wrong ones: ", wrong_ones)
print("right zeros: ", right_zeros, "wrong zeros: ", wrong_zeros)
print("result: right: ", right_ones + right_zeros, "||| wrong: ", wrong_ones + wrong_zeros)
print((right_ones + right_zeros) / len(y_test))

plt.show()

######################################################################################################
######################################################################################################
