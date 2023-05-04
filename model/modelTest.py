# Author: Saleh Shalabi
# Date: 2023-01-30

import tensorflow as tf
import tensorflow_io as tfio
import os

# Load the model
model_path = "../models/model.h5"
model = tf.keras.models.load_model(model_path)
config = model.get_config()

print(model.summary())
print(config["layers"][0]["config"]["batch_input_shape"])


def load_mp3_16k(file):
    """  Loads mp3 file and converts it to a 16kHz tensor """

    file_contents = tfio.audio.AudioIOTensor(file)

    tensor = file_contents.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2

    sample_rate = file_contents.rate

    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


@tf.function(reduce_retracing=True)
def preprocess(sample):
    """ Preprocesses the audio tensor and returns a spectrogram """

    spectrogram = tf.signal.stft(sample, frame_length=256, frame_step=64, window_fn=tf.signal.hamming_window)

    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.ensure_shape(spectrogram, [None, None, 1])
    spectrogram = tf.image.resize(spectrogram, [128, 64])
    spectrogram = tf.squeeze(spectrogram, axis=-1)

    spectrogram = spectrogram * 100
    spectrogram = tf.round(spectrogram)
    spectrogram = tf.math.divide_no_nan(spectrogram, 100)
    spectrogram = tf.cast(spectrogram, tf.float16)

    return spectrogram


def split_tensor_into_slices(tensor, slice_size, overlap_size=0):
    """
    Splits a tensor into overlapping slices of a given size
    Args:
        :param tensor: The tensor to split
        :param slice_size: The size of each slice
        :param overlap_size: The size of the overlap between slices
    Returns:
        A list of slices
    """

    # Get the number of slices
    num_slices = tensor.shape[0] // (slice_size - overlap_size)
    # Reshape the tensor into overlapping slices
    slices = []

    for sli in range(num_slices):

        start_index = sli * (slice_size - overlap_size)
        end_index = start_index + slice_size
        t = tensor[start_index:end_index]
        if t.shape[0] < slice_size:
            t = extend_tensor(t, slice_size)

        slices.append(t)

    return slices


def extend_tensor(tensor, size):
    """ Extends a tensor to a given size by adding zeros to the beginning """
    if tensor.shape[0] < size:
        pad = tf.zeros([size - tensor.shape[0]], dtype=tf.float32)
        tensor = tf.concat([pad, tensor], 0)
    return tensor


###################################################################################

length = 32000  # 2 seconds of audio
over_lap = 8000  # 0.5 seconds of overlap

dir_path = "../test_data/"  # test it with the positive or negative folder to be able to see the results
s = len(os.listdir(dir_path))
pos_res = 0
r = 0
for i in os.listdir(dir_path):

    r += 1
    wave = load_mp3_16k(os.path.join(dir_path, i))
    try:

        # if the audio is less than 2 seconds, pad it with zeros
        if wave.shape[0] < length:
            padding = tf.zeros([length - wave.shape[0]], dtype=tf.float32)
            wave = tf.concat([padding, wave], 0)

        # split the audio into overlapping slices to get a spectrogram for each slice and feed it to the model
        wave = split_tensor_into_slices(wave, slice_size=length, overlap_size=over_lap)
        audio_slices = tf.data.Dataset.from_tensor_slices(wave)

        audio_slices = audio_slices.map(preprocess)
        audio_slices = audio_slices.batch(32)
        yhat = model.predict(audio_slices, verbose=0)

        # if any slice predicts a positive result, the audio is considered positive
        yhat = [1 if i > 0.5 else 0 for i in yhat]

        if sum(yhat) > 0:
            pos_res += 1

        print(f"Tested {r} files")

    except Exception as e:
        print(i)
        print(e)

print(f"Total Tests: {r}")
print(f"Negative Result: {r - pos_res}")
print(f"Positive Result: {pos_res}")
