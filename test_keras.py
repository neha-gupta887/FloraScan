
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.layers import Dense
import json


# ============================================================
# KERAS COMPATIBILITY FIX
# ============================================================

_original_dense_from_config = Dense.from_config


@classmethod
def _compatible_dense_from_config(cls, config):
    config = dict(config)

    # Remove unsupported Keras parameter
    config.pop("quantization_config", None)

    return _original_dense_from_config(config)


Dense.from_config = _compatible_dense_from_config


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("model/trained_model.keras")

CLASS_NAMES_PATH = Path(
    "disease_data/class_names.json"
)

TEST_IMAGE = Path(
    r"C:\Users\Radha\OneDrive\Pictures\Screenshots\Screenshot 2026-10-21 224320.png"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FloraScan - Original Keras Model Test")
print("=" * 60)

print("\nLoading original Keras model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


# ============================================================
# CHECK IMAGE
# ============================================================

if not TEST_IMAGE.exists():
    print("\nERROR: Test image not found!")
    print(TEST_IMAGE)
    raise SystemExit


print("\nTesting image:")
print(TEST_IMAGE)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = Image.open(TEST_IMAGE).convert("RGB")

print("Original image size:", image.size)

image = image.resize((128, 128))

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = image_array / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)


print("Input array shape:", image_array.shape)
print("Input minimum:", float(np.min(image_array)))
print("Input maximum:", float(np.max(image_array)))


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning Keras prediction...")

predictions = model.predict(
    image_array,
    verbose=0
)

probabilities = np.asarray(predictions[0])


# ============================================================
# RAW OUTPUT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("RAW MODEL OUTPUT ANALYSIS")
print("=" * 60)

print("Output shape:", probabilities.shape)

print("Output sum:", float(np.sum(probabilities)))

print("Minimum output:", float(np.min(probabilities)))

print("Maximum output:", float(np.max(probabilities)))

print("=" * 60)


# ============================================================
# MAIN PREDICTION
# ============================================================

predicted_index = int(
    np.argmax(probabilities)
)

confidence = float(
    probabilities[predicted_index]
)

predicted_class = class_names[predicted_index]


print("\n" + "=" * 60)
print("KERAS MODEL PREDICTION")
print("=" * 60)

print("Class:", predicted_class)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)

print("=" * 60)


# ============================================================
# TOP 5 PREDICTIONS
# ============================================================

top_indices = np.argsort(probabilities)[-5:][::-1]

print("\nTOP 5 KERAS PREDICTIONS")
print("=" * 60)

for rank, index in enumerate(top_indices, start=1):

    print(
        f"{rank}. {class_names[index]} "
        f"-> {probabilities[index] * 100:.2f}%"
    )

print("=" * 60)