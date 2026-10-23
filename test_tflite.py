import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import json


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = Path("model/florascan_model.tflite")
CLASS_NAMES_PATH = Path("disease_data/class_names.json")

TEST_IMAGE = Path(
    r"C:\Users\Radha\OneDrive\Pictures\Screenshots\Screenshot 2026-10-21 224320.png"
)


# ------------------------------------------------------------
# Load TFLite model
# ------------------------------------------------------------

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details[0]["shape"])
print("Output:", output_details[0]["shape"])


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


# ------------------------------------------------------------
# Check test image
# ------------------------------------------------------------

if not TEST_IMAGE.exists():
    print("\nERROR: Test image not found!")
    print(TEST_IMAGE)
    exit()

print("\nTesting image:")
print(TEST_IMAGE)


# ------------------------------------------------------------
# Prepare image
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

interpreter.set_tensor(
    input_details[0]["index"],
    image_array
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

probabilities = output[0]


# ------------------------------------------------------------
# Prediction details
# ------------------------------------------------------------

predicted_index = int(
    np.argmax(probabilities)
)

confidence = float(
    probabilities[predicted_index]
)

predicted_class = class_names[predicted_index]


# ------------------------------------------------------------
# Output analysis
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("MODEL OUTPUT ANALYSIS")
print("=" * 50)

print("Output sum:", float(np.sum(probabilities)))
print("Maximum raw output:", confidence)

print("\n" + "=" * 50)
print("PREDICTION")
print("=" * 50)

print("Class:", predicted_class)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)

print("=" * 50)


# ------------------------------------------------------------
# Top 5 predictions
# ------------------------------------------------------------

top_indices = np.argsort(probabilities)[-5:][::-1]

print("\nTOP 5 PREDICTIONS")
print("=" * 50)

for rank, index in enumerate(top_indices, start=1):

    print(
        f"{rank}. {class_names[index]} "
        f"-> {probabilities[index] * 100:.2f}%"
    )

print("=" * 50)