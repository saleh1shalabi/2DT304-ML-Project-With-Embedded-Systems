# Author: Saleh Shalabi
# Date: 2023-01-30

import tensorflow as tf

# Load the frozen model
path_to_model = "../models/model.h5"
model = tf.keras.models.load_model(path_to_model)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimizations for the TFLite model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to a TFLite model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("../models/model.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted model to TFLite format")