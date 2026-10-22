import os
import io
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import keras
from keras.layers import Dense

import numpy as np
from PIL import Image


# ============================================================
# KERAS COMPATIBILITY FIX
# ============================================================

_original_dense_from_config = Dense.from_config


@classmethod
def _compatible_dense_from_config(cls, config):
    config = dict(config)

    # Older .keras models may contain this key.
    # Current Keras Dense does not accept it.
    config.pop("quantization_config", None)

    return _original_dense_from_config(config)


Dense.from_config = _compatible_dense_from_config


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "disease_data",
    "class_names.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.50")
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

ml_model = None
ml_class_names = []


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global ml_model, ml_class_names

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print(
            f"ERROR: Model file not found at {MODEL_PATH}"
        )

    else:

        try:

            print(
                f"Loading model from: {MODEL_PATH}"
            )

            ml_model = tf.keras.models.load_model(
                MODEL_PATH
            )

            print(
                f"SUCCESS: Model loaded from {MODEL_PATH}"
            )

        except Exception as e:

            print(
                f"ERROR loading model: {e}"
            )

    # --------------------------------------------------------
    # Load Class Names
    # --------------------------------------------------------

    if not os.path.exists(CLASS_NAMES_PATH):

        print(
            f"ERROR: Class names file not found at "
            f"{CLASS_NAMES_PATH}"
        )

    else:

        try:

            with open(
                CLASS_NAMES_PATH,
                "r"
            ) as f:

                ml_class_names = json.load(f)

            print(
                f"SUCCESS: Loaded "
                f"{len(ml_class_names)} class names."
            )

        except Exception as e:

            print(
                f"ERROR loading class names: {e}"
            )

    # --------------------------------------------------------
    # Validate Model + Classes
    # --------------------------------------------------------

    if ml_model is not None and ml_class_names:

        if len(ml_class_names) != 38:

            print(
                f"ERROR: Expected 38 classes, "
                f"found {len(ml_class_names)}."
            )

        try:

            model_output_classes = (
                ml_model.output_shape[-1]
            )

            if (
                len(ml_class_names)
                != model_output_classes
            ):

                print(
                    f"ERROR: Class names length "
                    f"({len(ml_class_names)}) "
                    f"does not match model output "
                    f"({model_output_classes})."
                )

        except Exception as e:

            print(
                f"WARNING: Could not validate "
                f"model output shape: {e}"
            )

    yield

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    ml_model = None
    ml_class_names.clear()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Plant Disease ML Service",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():

    is_loaded = (
        ml_model is not None
        and len(ml_class_names) > 0
    )

    if not is_loaded:

        return {
            "status": "unhealthy",
            "modelLoaded": False,
            "classCount": len(ml_class_names)
        }

    return {
        "status": "healthy",
        "modelLoaded": True,
        "classCount": len(ml_class_names)
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate File
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="File provided is not an image."
        )

    # --------------------------------------------------------
    # Check Model
    # --------------------------------------------------------

    if (
        ml_model is None
        or not ml_class_names
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded properly."
        )

    try:

        # ----------------------------------------------------
        # Read Image
        # ----------------------------------------------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        if image.mode != "RGB":

            image = image.convert("RGB")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            (128, 128)
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        input_arr = np.array(image)

        # Handle RGBA images
        if (
            len(input_arr.shape) == 3
            and input_arr.shape[-1] == 4
        ):

            input_arr = input_arr[:, :, :3]

        # Add batch dimension
        input_arr = np.expand_dims(
            input_arr,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = ml_model.predict(
            input_arr,
            verbose=0
        )

        # ----------------------------------------------------
        # Top 3 Predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            prediction[0]
        )[-3:][::-1]

        top_predictions = []

        for i in top_indices:

            top_predictions.append(
                {
                    "className": ml_class_names[i],
                    "confidence": float(
                        prediction[0][i]
                    )
                }
            )

        # ----------------------------------------------------
        # Best Prediction
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(prediction[0])
        )

        max_confidence = float(
            prediction[0][predicted_index]
        )

        predicted_class = (
            ml_class_names[predicted_index]
        )

        # ----------------------------------------------------
        # Confidence Check
        # ----------------------------------------------------

        if (
            max_confidence
            < PREDICTION_CONFIDENCE_THRESHOLD
        ):

            return JSONResponse(
                content={
                    "status": "uncertain",

                    "message":
                        "Unable to confidently "
                        "identify a plant disease "
                        "from this image.",

                    "confidence":
                        max_confidence,

                    "topPredictions":
                        top_predictions
                },

                status_code=200
            )

        # ----------------------------------------------------
        # Successful Prediction
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "status": "success",

                "predictedClass":
                    predicted_class,

                "confidence":
                    max_confidence,

                "topPredictions":
                    top_predictions
            },

            status_code=200
        )

    # --------------------------------------------------------
    # Prediction Error
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return JSONResponse(
            status_code=500,

            content={
                "error":
                    "Prediction service failed "
                    "while analyzing the image."
            }
        )