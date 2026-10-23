import os
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import Dense


# ============================================================
# KERAS COMPATIBILITY FIX
# ============================================================

_original_dense_from_config = Dense.from_config


@classmethod
def _compatible_dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)


Dense.from_config = _compatible_dense_from_config


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("model/trained_model.keras")
OUTPUT_PATH = Path("model/florascan_model.tflite")


print("=" * 60)
print("FloraScan - Keras to TFLite Conversion")
print("=" * 60)


try:

    print("\n[1/4] Checking model...")

    print("Model exists:", MODEL_PATH.exists())
    print("Model path:", MODEL_PATH.resolve())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )


    print("\n[2/4] Loading Keras model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)


    print("\n[3/4] Converting to TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    # No quantization
    tflite_model = converter.convert()

    print("Conversion completed!")
    print("TFLite bytes:", len(tflite_model))


    print("\n[4/4] Saving model...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)


    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)

    print("Output:", OUTPUT_PATH.resolve())

    print(
        "Model size:",
        round(
            OUTPUT_PATH.stat().st_size / (1024 * 1024),
            2
        ),
        "MB"
    )

    print("=" * 60)


except Exception as e:

    print("\n" + "=" * 60)
    print("CONVERSION FAILED")
    print("=" * 60)

    print("Error type:", type(e).__name__)
    print("Error:", str(e))

    print("\nFull traceback:")

    traceback.print_exc()