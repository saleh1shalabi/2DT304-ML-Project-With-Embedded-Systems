# Author: Saleh Shalabi
# Date: 2023-02-03

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_io as tfio


def preprocess(wave):
    """ Preprocesses the audio tensor and returns a spectrogram """
    wave = tf.squeeze(wave, axis=-1)
    wave = tfio.audio.resample(wave, rate_in=44100, rate_out=16000)

    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=64, window_fn=tf.signal.hamming_window,
                                 pad_end=True)

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


def record_audio(time, sample_rate):
    """ Records audio from the microphone for a given time and sample rate """
    recording = sd.rec(frames=int(time * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    return recording


print("Choose the model to use: ")
print("1. Usual model")
print("2. TfLite model")

model_choice = int(input("Enter the number of the model to use: "))

if model_choice == 1:
    model = tf.keras.models.load_model('../models/model100epochs.h5')
    print(model.summary())
    print(model.get_config()["layers"][0]["config"]["batch_input_shape"])
elif model_choice == 2:
    interpreter = tf.lite.Interpreter(model_path="../models/model.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    print(interpreter.get_input_details())
else:
    print("Invalid choice")
    exit()

duration = 2  # seconds

devices = sd.query_devices()
print("Available microphones:")
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']}")

# Record from a specific microphone
microphone_index = int(input("Enter the index of the microphone to use: "))

try:
    sd.default.device = microphone_index
except Exception as e:
    print("Invalid microphone index")
    exit()

if model_choice == 1:

    while True:
        print("recording...")
        my_recording = record_audio(duration, 16000)
        print("recording done")

        # Preprocess the audio
        v = preprocess(my_recording)
        s = tf.expand_dims(v, axis=0)

        x = model.predict(s)

        print(f"Predicted value: {x}")

else:
    while True:
        print("recording...")

        my_recording = record_audio(duration, 16000)
        print("recording done")

        # Preprocess the audio
        v = preprocess(my_recording)
        v = tf.reshape(v, [-1, 128, 64])

        input_data = v.numpy().astype(np.float32)

        # Use the TFLite interpreter to make predictions
        interpreter.set_tensor(input_index, v)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        print(f"Predicted value: {output_data}")
