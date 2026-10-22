import os

# ============================================================
# ENVIRONMENT / MEMORY SETTINGS
# ============================================================

# Reduce TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Render par GPU available nahi hai
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================
# IMPORTS
# ============================================================

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from tensorflow.keras.layers import Dense


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FloraScan AI API",
    description="Plant disease detection API powered by TensorFlow",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

frontend_url = os.getenv("FRONTEND_URL", "*")

if frontend_url == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        frontend_url,
        frontend_url.rstrip("/")
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

# ml-service/
ML_SERVICE_DIR = Path(__file__).resolve().parent

# Project root/
PROJECT_ROOT = ML_SERVICE_DIR.parent

# Possible local model locations
LOCAL_MODEL_PATHS = [
    PROJECT_ROOT / "model" / "trained_model.keras",
    PROJECT_ROOT / "trained_model.keras",
]

# Cached downloaded model
CACHE_DIR = ML_SERVICE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHED_MODEL_PATH = CACHE_DIR / "trained_model.keras"

# Hugging Face model
DEFAULT_MODEL_URL = (
    "https://huggingface.co/nehag65/florascan-model/"
    "resolve/main/trained_model.keras?download=true"
)

MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)

# Class names
CLASS_NAMES_PATH = (
    PROJECT_ROOT / "disease_data" / "class_names.json"
)


# ============================================================
# MODEL VARIABLES
# ============================================================

model = None
class_names = []


# ============================================================
# KERAS COMPATIBILITY FIX
# ============================================================

# Some saved Keras models contain:
# "quantization_config": None
#
# Older/current Dense implementation can reject this argument.
# Remove it before Dense configuration is created.

_original_dense_from_config = Dense.from_config


@classmethod
def _compatible_dense_from_config(cls, config):
    config = dict(config)

    config.pop("quantization_config", None)

    return _original_dense_from_config(config)


Dense.from_config = _compatible_dense_from_config


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    global class_names

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json not found at: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both:
    # ["Apple___Apple_scab", ...]
    #
    # and:
    # {"0": "Apple___Apple_scab", ...}

    if isinstance(data, list):
        class_names = data

    elif isinstance(data, dict):
        # Sort numeric keys if possible
        try:
            sorted_items = sorted(
                data.items(),
                key=lambda item: int(item[0])
            )
            class_names = [value for _, value in sorted_items]
        except (ValueError, TypeError):
            class_names = list(data.values())

    else:
        raise ValueError(
            "Unsupported class_names.json format."
        )

    if not class_names:
        raise ValueError(
            "class_names.json is empty."
        )


# ============================================================
# FIND / DOWNLOAD MODEL
# ============================================================

def get_model_path():
    # --------------------------------------------------------
    # 1. Check local model
    # --------------------------------------------------------

    for path in LOCAL_MODEL_PATHS:
        if path.exists():
            print(f"Using local model: {path}")
            return path

    # --------------------------------------------------------
    # 2. Check cached model
    # --------------------------------------------------------

    if CACHED_MODEL_PATH.exists():
        print(
            f"Using cached Hugging Face model: "
            f"{CACHED_MODEL_PATH}"
        )
        return CACHED_MODEL_PATH

    # --------------------------------------------------------
    # 3. Download from Hugging Face
    # --------------------------------------------------------

    print("Local model not found.")
    print("Downloading model from Hugging Face...")
    print(f"Model URL: {MODEL_URL}")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            CACHED_MODEL_PATH
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from Hugging Face: {e}"
        )

    if not CACHED_MODEL_PATH.exists():
        raise RuntimeError(
            "Model download completed but file was not found."
        )

    print(
        f"Model downloaded successfully: "
        f"{CACHED_MODEL_PATH}"
    )

    return CACHED_MODEL_PATH


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global model

    model_path = get_model_path()

    print("Loading TensorFlow/Keras model...")
    print(f"Model path: {model_path}")

    # IMPORTANT:
    # compile=False avoids unnecessary optimizer/training
    # objects and reduces memory usage during startup.

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print("Model loaded successfully.")

    # Validate output classes
    try:
        output_shape = model.output_shape

        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        model_class_count = output_shape[-1]

        print(
            f"Model output classes: "
            f"{model_class_count}"
        )

        print(
            f"Class names count: "
            f"{len(class_names)}"
        )

        if model_class_count != len(class_names):
            raise ValueError(
                f"Model has {model_class_count} outputs, "
                f"but class_names.json has "
                f"{len(class_names)} classes."
            )

    except Exception as e:
        print(
            f"Warning while validating model output: {e}"
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("Starting FloraScan AI API")
    print("=" * 60)

    try:
        load_class_names()
        print(
            f"Loaded {len(class_names)} class names."
        )

        load_model()

        print("=" * 60)
        print("FloraScan AI API is ready!")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("MODEL STARTUP ERROR")
        print("=" * 60)
        print(str(e))
        print("=" * 60)

        # Keep server alive so /health can report failure
        model = None


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "FloraScan AI API",
        "status": "running",
        "modelLoaded": model is not None
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "modelLoaded": model is not None,
        "classCount": len(class_names)
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type."
        )

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a JPG, JPEG, PNG, "
                "or WEBP image."
            )
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:
        image_bytes = await file.read()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        )

        image = image.convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize((128, 128))

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = image_array / 255.0

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:
        predictions = model.predict(
            image_array,
            verbose=0
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = np.asarray(
        predictions[0]
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Get disease/class name
    # --------------------------------------------------------

    if predicted_index >= len(class_names):
        raise HTTPException(
            status_code=500,
            detail="Predicted class index is invalid."
        )

    predicted_class = class_names[
        predicted_index
    ]

    # --------------------------------------------------------
    # Split plant and disease
    #
    # Example:
    # Tomato___Late_blight
    # --------------------------------------------------------

    if "___" in predicted_class:

        plant_name, disease_name = (
            predicted_class.split(
                "___",
                1
            )
        )

    else:

        plant_name = predicted_class
        disease_name = predicted_class

    # Make names user-friendly
    plant_name = plant_name.replace("_", " ")
    disease_name = disease_name.replace("_", " ")

    # --------------------------------------------------------
    # Return only MAIN diagnosis
    # --------------------------------------------------------

    return {
        "plant": plant_name,
        "condition": disease_name,
        "confidence": confidence
    }