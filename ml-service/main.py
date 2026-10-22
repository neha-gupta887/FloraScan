import os
import json
import urllib.request
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import ai_edge_litert.interpreter as tflite
# ============================================================
# CONFIG
# ============================================================

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/nehag65/florascan-model/"
    "resolve/main/florascan_model.tflite?download=true"
)

ML_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_SERVICE_DIR.parent

CACHE_DIR = ML_SERVICE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CACHE_DIR / "florascan_model.tflite"

CLASS_NAMES_PATH = (
    PROJECT_ROOT / "disease_data" / "class_names.json"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="FloraScan AI API",
    description="Lightweight plant disease detection API",
    version="2.0.0"
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
# GLOBALS
# ============================================================

interpreter = None
class_names = []


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    global class_names

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json not found: "
            f"{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, list):

        class_names = data

    elif isinstance(data, dict):

        try:

            sorted_items = sorted(
                data.items(),
                key=lambda item: int(item[0])
            )

            class_names = [
                value
                for _, value in sorted_items
            ]

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
# DOWNLOAD MODEL
# ============================================================

def download_model():

    if MODEL_PATH.exists():

        print(
            f"Using cached TFLite model: {MODEL_PATH}"
        )

        return

    print("TFLite model not found locally.")
    print("Downloading from Hugging Face...")
    print(f"URL: {MODEL_URL}")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to download TFLite model: {e}"
        )

    if not MODEL_PATH.exists():

        raise RuntimeError(
            "Model download failed."
        )

    print(
        f"Model downloaded successfully: {MODEL_PATH}"
    )


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

def load_model():

    global interpreter

    download_model()

    print("Loading TFLite interpreter...")

    interpreter = tflite.Interpreter(
    model_path=str(MODEL_PATH),
    num_threads=2
)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(
        "Input shape:",
        input_details[0]["shape"]
    )

    print(
        "Output shape:",
        output_details[0]["shape"]
    )

    # Validate output classes

    output_shape = output_details[0]["shape"]

    model_class_count = int(
        output_shape[-1]
    )

    if model_class_count != len(class_names):

        raise ValueError(
            f"Model has {model_class_count} classes "
            f"but class_names.json has "
            f"{len(class_names)} classes."
        )

    print(
        f"TFLite model loaded successfully "
        f"with {model_class_count} classes."
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global interpreter

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
        print("FloraScan AI API is READY!")
        print("=" * 60)

    except Exception as e:

        print("=" * 60)
        print("STARTUP ERROR")
        print("=" * 60)

        print(str(e))

        print("=" * 60)

        interpreter = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "FloraScan AI API",
        "version": "2.0.0",
        "status": "running",
        "modelLoaded": interpreter is not None
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if interpreter is not None
            else "unhealthy"
        ),
        "modelLoaded": interpreter is not None,
        "classCount": len(class_names)
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if interpreter is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
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
                "Please upload a JPG, JPEG, "
                "PNG, or WEBP image."
            )
        )

    try:

        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = image.convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image."
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(
        (128, 128)
    )

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # TFLite input/output
    # --------------------------------------------------------

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_index = (
        input_details[0]["index"]
    )

    output_index = (
        output_details[0]["index"]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        interpreter.set_tensor(
            input_index,
            image_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_index
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Main prediction
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

    predicted_class = class_names[
        predicted_index
    ]

    # --------------------------------------------------------
    # Plant + condition
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

    plant_name = plant_name.replace(
        "_",
        " "
    )

    disease_name = disease_name.replace(
        "_",
        " "
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "plant": plant_name,
        "condition": disease_name,
        "confidence": confidence
    }