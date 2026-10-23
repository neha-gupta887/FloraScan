import os
import json
import urllib.request
from pathlib import Path
from io import BytesIO
from contextlib import asynccontextmanager

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

# Confidence below this value is treated as uncertain.
# Keep this configurable through Render environment variables.
PREDICTION_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "PREDICTION_CONFIDENCE_THRESHOLD",
        "0.50"
    )
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

            class_names = list(
                data.values()
            )

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
# LOAD TFLITE / LITERT MODEL
# ============================================================

def load_model():

    global interpreter

    download_model()

    print("Loading LiteRT/TFLite interpreter...")

    interpreter = tflite.Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    print(
        "Input shape:",
        input_details[0]["shape"]
    )

    print(
        "Input dtype:",
        input_details[0]["dtype"]
    )

    print(
        "Output shape:",
        output_details[0]["shape"]
    )

    print(
        "Output dtype:",
        output_details[0]["dtype"]
    )

    # --------------------------------------------------------
    # Validate output classes
    # --------------------------------------------------------

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
        f"LiteRT/TFLite model loaded successfully "
        f"with {model_class_count} classes."
    )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

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

    yield

    print("FloraScan AI API shutting down...")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="FloraScan AI API",
    description="Lightweight plant disease detection API",
    version="2.3.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

frontend_url = os.getenv(
    "FRONTEND_URL",
    "*"
)

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
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "FloraScan AI API",
        "version": "2.3.0",
        "status": "running",
        "modelLoaded": interpreter is not None,
        "classCount": len(class_names)
    }


# ============================================================
# HEALTH CHECK
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
# FORMAT CLASS NAME
# ============================================================

def format_prediction(class_name):

    if "___" in class_name:

        plant_name, disease_name = (
            class_name.split(
                "___",
                1
            )
        )

    else:

        plant_name = class_name
        disease_name = class_name

    plant_name = plant_name.replace(
        "_",
        " "
    ).strip()

    disease_name = disease_name.replace(
        "_",
        " "
    ).strip()

    return plant_name, disease_name


# ============================================================
# CHECK HEALTHY
# ============================================================

def check_health_status(class_name):

    is_healthy = (
        "healthy" in class_name.lower()
    )

    health_status = (
        "Healthy"
        if is_healthy
        else "Diseased"
    )

    return is_healthy, health_status


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Model input size
    image = image.resize(
        (128, 128)
    )

    # IMPORTANT:
    # Do NOT divide by 255.
    #
    # Original FloraScan preprocessing used
    # raw pixel values from 0-255.

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if interpreter is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # FILE TYPE CHECK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = image.convert("RGB")

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read uploaded image: "
                + str(e)
            )
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_array = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # INPUT / OUTPUT DETAILS
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

    expected_dtype = (
        input_details[0]["dtype"]
    )

    # --------------------------------------------------------
    # MATCH MODEL DTYPE
    # --------------------------------------------------------

    if image_array.dtype != expected_dtype:

        image_array = image_array.astype(
            expected_dtype
        )

    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    try:

        interpreter.set_tensor(
            input_index,
            image_array
        )

        interpreter.invoke()

        predictions = (
            interpreter.get_tensor(
                output_index
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # GET PREDICTIONS
    # --------------------------------------------------------

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # VALIDATE CLASS COUNT
    # --------------------------------------------------------

    if len(probabilities) != len(class_names):

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model returned "
                f"{len(probabilities)} predictions "
                f"but there are "
                f"{len(class_names)} class names."
            )
        )

    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top_predictions = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        index = int(index)

        confidence = float(
            probabilities[index]
        )

        predicted_class = (
            class_names[index]
        )

        plant_name, disease_name = (
            format_prediction(
                predicted_class
            )
        )

        prediction_item = {

            "rank": rank,

            "plant": plant_name,

            "condition": disease_name,

            "confidence": confidence
        }

        top_predictions.append(
            prediction_item
        )

    # --------------------------------------------------------
    # BEST PREDICTION
    # --------------------------------------------------------

    best_index = int(
        np.argmax(probabilities)
    )

    best_class = (
        class_names[best_index]
    )

    best_confidence = float(
        probabilities[best_index]
    )

    best_plant, best_condition = (
        format_prediction(
            best_class
        )
    )

    # --------------------------------------------------------
    # HEALTH STATUS
    # --------------------------------------------------------

    is_healthy, health_status = (
        check_health_status(
            best_class
        )
    )

    # --------------------------------------------------------
    # UNCERTAIN PREDICTION
    # --------------------------------------------------------
    #
    # Example:
    # Tulsi image is outside the model's known classes.
    #
    # If model confidence is only 6%, don't confidently
    # tell the user it is Tomato.
    #
    # --------------------------------------------------------

    if (
        best_confidence
        < PREDICTION_CONFIDENCE_THRESHOLD
    ):

        return {

            "status": "uncertain",

            "message": (
                "Unable to confidently identify "
                "a plant disease from this image."
            ),

            "plant": None,

            "condition": None,

            "confidence": best_confidence,

            "is_healthy": None,

            "health_status": "Unknown",

            "top_predictions": top_predictions
        }

    # --------------------------------------------------------
    # SUCCESS RESPONSE
    # --------------------------------------------------------

    return {

        "status": "success",

        "plant": best_plant,

        "condition": best_condition,

        "confidence": best_confidence,

        "is_healthy": is_healthy,

        "health_status": health_status,

        "top_predictions": top_predictions
    }