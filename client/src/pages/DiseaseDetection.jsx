import React, { useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Leaf,
  UploadCloud,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  Sun,
  Camera,
  Ban,
  Info,
  ArrowRight,
  ScanLine,
} from "lucide-react";

const API_URL = "https://florascan-api-y534.onrender.com";
const DiseaseDetection = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");

  // ----------------------------------------
  // FILE VALIDATION
  // ----------------------------------------
  const validateFile = (selectedFile) => {
    if (!selectedFile) return false;

    const allowedTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please upload a JPG, JPEG, PNG, or WEBP image.");
      return false;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return false;
    }

    return true;
  };

  // ----------------------------------------
  // HANDLE FILE
  // ----------------------------------------
  const handleFile = (selectedFile) => {
    if (!validateFile(selectedFile)) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  // ----------------------------------------
  // INPUT CHANGE
  // ----------------------------------------
  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];

    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  // ----------------------------------------
  // DRAG EVENTS
  // ----------------------------------------
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files?.[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  };

  // ----------------------------------------
  // ANALYZE IMAGE
  // ----------------------------------------
  const analyzeLeaf = async () => {
    if (!file) {
      setError("Please upload a leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      // FastAPI expects the field name "file"
      formData.append("file", file);

      const response = await axios.post(
        `${API_URL}/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 120000,
        }
      );

      console.log("FloraScan API Response:", response.data);

      const data = response.data;

      // ----------------------------------------
      // GET PREDICTION FROM FASTAPI
      // Backend response:
      // {
      //   "plant": "Tomato",
      //   "condition": "Late blight",
      //   "confidence": 0.0566
      // }
      // ----------------------------------------
      const plantFromAPI = data.plant || "";
      const diseaseFromAPI = data.condition || "";

      if (!diseaseFromAPI) {
        throw new Error("Prediction was not returned by the AI model.");
      }

      // ----------------------------------------
      // PLANT
      // ----------------------------------------
      const plant =
        plantFromAPI
          .replace(/_/g, " ")
          .replace(/\b\w/g, (char) => char.toUpperCase()) ||
        "Unknown Plant";

      // ----------------------------------------
      // DISEASE / CONDITION
      // ----------------------------------------
      const disease = diseaseFromAPI
        .replace(/_/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());

      // ----------------------------------------
      // CONFIDENCE
      // ----------------------------------------
      let rawConfidence = data.confidence;

      let confidence = Number(rawConfidence);

      if (Number.isNaN(confidence)) {
        confidence = 0;
      }

      /*
        Backend returns confidence as decimal:

        0.0566 -> 5.66%

        If backend ever returns:
        5.66 -> 5.66%

        We handle both safely.
      */
      if (confidence >= 0 && confidence <= 1) {
        confidence = confidence * 100;
      }

      // Keep confidence between 0 and 100
      confidence = Math.min(100, Math.max(0, confidence));

      // ----------------------------------------
      // SAVE RESULT
      // ----------------------------------------
      setResult({
        plant,
        disease,
        confidence,
      });
    } catch (err) {
      console.error("FloraScan prediction error:", err);

      if (err.response) {
        console.error("Backend response:", err.response.data);
      }

      if (err.code === "ECONNABORTED") {
        setError(
          "The AI service took too long to respond. Please try again."
        );
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.message) {
        setError(err.message);
      } else {
        setError(
          "Unable to analyze the image. Please make sure the AI service is running."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------
  // RESET
  // ----------------------------------------
  const resetScanner = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setLoading(false);
  };

  // ----------------------------------------
  // CONFIDENCE LABEL
  // ----------------------------------------
  const getConfidenceLabel = (confidence) => {
    if (confidence >= 80) return "High Confidence";
    if (confidence >= 60) return "Good Confidence";
    if (confidence >= 40) return "Moderate Confidence";
    return "Low Confidence";
  };

  return (
    <div className="min-h-screen bg-white text-slate-900">

      {/* ======================================
          NAVBAR
      ====================================== */}
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">

          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-600 shadow-lg shadow-emerald-600/20">
              <Leaf className="h-6 w-6 text-white" />
            </div>

            <div>
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900">
                Flora<span className="text-emerald-600">Scan</span>
              </h1>

              <p className="hidden text-[9px] font-semibold tracking-[0.2em] text-slate-400 sm:block">
                AI-POWERED PLANT HEALTH
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-8 text-sm font-semibold text-slate-600 md:flex">
            <a href="#" className="transition hover:text-emerald-600">
              Home
            </a>

            <a
              href="#scanner"
              className="transition hover:text-emerald-600"
            >
              Scanner
            </a>

            <a
              href="#tips"
              className="transition hover:text-emerald-600"
            >
              Tips
            </a>
          </div>

          <div className="hidden rounded-full bg-emerald-50 px-4 py-2 text-xs font-bold text-emerald-700 sm:block">
            Scan • Detect • Protect
          </div>
        </div>
      </nav>

      {/* ======================================
          HERO
      ====================================== */}
      <section className="overflow-hidden bg-gradient-to-br from-emerald-50 via-white to-lime-50">
        <div className="mx-auto grid max-w-7xl items-center gap-12 px-5 py-16 sm:px-8 lg:grid-cols-2 lg:py-24">

          <motion.div
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-white px-4 py-2 text-xs font-bold text-emerald-700 shadow-sm">
              <Sparkles className="h-4 w-4" />
              AI-POWERED PLANT DISEASE DETECTION
            </div>

            <h2 className="max-w-3xl text-4xl font-black leading-tight tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              Scan your leaf.
              <br />
              <span className="text-emerald-600">
                Understand its health.
              </span>
            </h2>

            <p className="mt-6 max-w-xl text-base leading-7 text-slate-600 sm:text-lg">
              Upload a clear image of a plant leaf and let FloraScan's
              AI-powered system identify possible diseases within seconds.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <div className="flex items-center gap-2 rounded-xl bg-white px-4 py-3 text-sm font-semibold shadow-sm ring-1 ring-slate-100">
                <Sparkles className="h-4 w-4 text-emerald-600" />
                AI Analysis
              </div>

              <div className="flex items-center gap-2 rounded-xl bg-white px-4 py-3 text-sm font-semibold shadow-sm ring-1 ring-slate-100">
                <ShieldCheck className="h-4 w-4 text-emerald-600" />
                Easy to Use
              </div>

              <div className="flex items-center gap-2 rounded-xl bg-white px-4 py-3 text-sm font-semibold shadow-sm ring-1 ring-slate-100">
                <Leaf className="h-4 w-4 text-emerald-600" />
                Plant Focused
              </div>
            </div>
          </motion.div>

          {/* Hero visual */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7 }}
            className="flex justify-center"
          >
            <div className="relative flex h-72 w-72 items-center justify-center sm:h-96 sm:w-96">

              <div className="absolute inset-0 rounded-full border border-emerald-200" />

              <motion.div
                animate={{ rotate: 360 }}
                transition={{
                  duration: 18,
                  repeat: Infinity,
                  ease: "linear",
                }}
                className="absolute inset-7 rounded-full border border-dashed border-emerald-300"
              />

              <div className="flex h-40 w-40 items-center justify-center rounded-full bg-emerald-600 shadow-2xl shadow-emerald-600/30 sm:h-48 sm:w-48">
                <Leaf className="h-20 w-20 text-white sm:h-24 sm:w-24" />
              </div>

              <div className="absolute right-3 top-12 rounded-2xl bg-white p-4 shadow-xl sm:right-8">
                <ScanLine className="h-6 w-6 text-emerald-600" />
              </div>

              <div className="absolute bottom-12 left-3 rounded-2xl bg-white p-4 shadow-xl sm:left-8">
                <CheckCircle2 className="h-6 w-6 text-emerald-600" />
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ======================================
          SCANNER
      ====================================== */}
      <section
        id="scanner"
        className="bg-slate-50 px-5 py-16 sm:px-8 lg:py-24"
      >
        <div className="mx-auto max-w-7xl">

          <div className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-600">
              AI Scanner
            </p>

            <h3 className="mt-3 text-3xl font-black tracking-tight text-slate-900 sm:text-4xl">
              Analyze your plant leaf
            </h3>

            <p className="mt-4 text-slate-600">
              Upload a clear image and FloraScan will analyze it using the
              trained AI model.
            </p>
          </div>

          <div className="grid gap-7 lg:grid-cols-2">

            {/* ==================================
                UPLOAD CARD
            ================================== */}
            <motion.div
              layout
              className="rounded-3xl bg-white p-5 shadow-xl shadow-slate-200/50 ring-1 ring-slate-200 sm:p-7"
            >
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h4 className="text-xl font-extrabold text-slate-900">
                    Upload Leaf
                  </h4>

                  <p className="mt-1 text-sm text-slate-500">
                    JPG, PNG, JPEG or WEBP
                  </p>
                </div>

                <div className="rounded-xl bg-emerald-50 p-3">
                  <Camera className="h-5 w-5 text-emerald-600" />
                </div>
              </div>

              {!preview ? (
                <label
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`flex min-h-[350px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed transition ${
                    dragActive
                      ? "border-emerald-500 bg-emerald-50"
                      : "border-slate-200 bg-slate-50 hover:border-emerald-300 hover:bg-emerald-50/40"
                  }`}
                >
                  <input
                    type="file"
                    accept="image/jpeg,image/jpg,image/png,image/webp"
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-100">
                    <UploadCloud className="h-8 w-8 text-emerald-600" />
                  </div>

                  <p className="text-center text-base font-bold text-slate-800">
                    Drag & drop your leaf image
                  </p>

                  <p className="mt-2 text-sm text-slate-500">
                    or click to browse from your device
                  </p>

                  <div className="mt-6 rounded-full bg-white px-4 py-2 text-xs font-semibold text-slate-500 shadow-sm">
                    Maximum size: 10 MB
                  </div>
                </label>
              ) : (
                <div className="relative overflow-hidden rounded-2xl bg-slate-100">
                  <img
                    src={preview}
                    alt="Uploaded plant leaf"
                    className="h-[350px] w-full object-contain"
                  />

                  <button
                    onClick={resetScanner}
                    className="absolute right-4 top-4 flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-sm font-bold text-slate-700 shadow-lg transition hover:bg-slate-50"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Change
                  </button>
                </div>
              )}

              {file && (
                <div className="mt-4 flex items-center gap-3 rounded-xl bg-emerald-50 p-4">
                  <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600" />

                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold text-slate-800">
                      {file.name}
                    </p>

                    <p className="text-xs text-slate-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              )}

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mt-5 flex items-start gap-3 rounded-xl bg-red-50 p-4 text-sm text-red-700"
                  >
                    <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <button
                onClick={analyzeLeaf}
                disabled={!file || loading}
                className="mt-6 flex w-full items-center justify-center gap-3 rounded-2xl bg-emerald-600 px-6 py-4 text-base font-extrabold text-white shadow-lg shadow-emerald-600/20 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Analyzing Leaf...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    Analyze Leaf
                    <ArrowRight className="h-5 w-5" />
                  </>
                )}
              </button>
            </motion.div>

            {/* ==================================
                RESULT CARD
            ================================== */}
            <motion.div
              layout
              className="rounded-3xl bg-white p-5 shadow-xl shadow-slate-200/50 ring-1 ring-slate-200 sm:p-7"
            >
              <div className="mb-6">
                <h4 className="text-xl font-extrabold text-slate-900">
                  AI Diagnosis
                </h4>

                <p className="mt-1 text-sm text-slate-500">
                  Your AI-powered plant health result appears here.
                </p>
              </div>

              {!result && !loading && (
                <div className="flex min-h-[500px] flex-col items-center justify-center rounded-2xl bg-slate-50 text-center">
                  <div className="mb-5 flex h-20 w-20 items-center justify-center rounded-full bg-emerald-100">
                    <Leaf className="h-10 w-10 text-emerald-600" />
                  </div>

                  <h5 className="text-lg font-extrabold text-slate-800">
                    Ready to scan
                  </h5>

                  <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                    Upload a leaf image on the left and click{" "}
                    <span className="font-semibold text-slate-700">
                      Analyze Leaf
                    </span>{" "}
                    to see the AI prediction.
                  </p>
                </div>
              )}

              {loading && (
                <div className="flex min-h-[500px] flex-col items-center justify-center rounded-2xl bg-slate-50">
                  <div className="relative flex h-24 w-24 items-center justify-center">
                    <div className="absolute inset-0 animate-ping rounded-full bg-emerald-100" />

                    <div className="relative flex h-20 w-20 items-center justify-center rounded-full bg-emerald-600">
                      <ScanLine className="h-9 w-9 text-white" />
                    </div>
                  </div>

                  <h5 className="mt-7 text-lg font-extrabold text-slate-800">
                    Analyzing your leaf...
                  </h5>

                  <p className="mt-2 text-sm text-slate-500">
                    FloraScan AI is processing the image.
                  </p>
                </div>
              )}

              <AnimatePresence>
                {result && !loading && (
                  <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >

                    {/* Detection complete */}
                    <div className="flex items-center gap-4 rounded-2xl bg-amber-50 p-5">
                      <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-orange-500">
                        <Leaf className="h-7 w-7 text-white" />
                      </div>

                      <div>
                        <h5 className="font-extrabold text-slate-800">
                          Detection Complete
                        </h5>

                        <p className="text-sm text-slate-500">
                          AI prediction generated successfully
                        </p>
                      </div>
                    </div>

                    {/* Plant */}
                    <div className="rounded-2xl bg-slate-50 p-6">
                      <p className="text-xs font-extrabold uppercase tracking-[0.15em] text-slate-400">
                        Plant
                      </p>

                      <p className="mt-3 text-3xl font-black text-slate-900">
                        {result.plant}
                      </p>
                    </div>

                    {/* Disease */}
                    <div className="rounded-2xl bg-slate-50 p-6">
                      <p className="text-xs font-extrabold uppercase tracking-[0.15em] text-slate-400">
                        Detected Condition
                      </p>

                      <p className="mt-3 text-2xl font-black text-orange-600">
                        {result.disease}
                      </p>
                    </div>

                    {/* Confidence */}
                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-extrabold text-slate-700">
                            AI Confidence
                          </p>

                          <p className="mt-1 text-xs text-slate-400">
                            {getConfidenceLabel(result.confidence)}
                          </p>
                        </div>

                        <p className="text-2xl font-black text-emerald-600">
                          {result.confidence.toFixed(2)}%
                        </p>
                      </div>

                      {/* Progress bar */}
                      <div className="mt-5 h-3 overflow-hidden rounded-full bg-slate-100">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{
                            width: `${result.confidence}%`,
                          }}
                          transition={{
                            duration: 0.8,
                            ease: "easeOut",
                          }}
                          className="h-full rounded-full bg-emerald-500"
                        />
                      </div>
                    </div>

                    {/* Disclaimer */}
                    <div className="flex items-start gap-3 rounded-2xl bg-emerald-50 p-5 text-sm leading-6 text-emerald-800">
                      <Info className="mt-0.5 h-5 w-5 shrink-0" />

                      <p>
                        This is an AI-generated prediction. For important
                        agricultural decisions, verify the result with a
                        qualified expert.
                      </p>
                    </div>

                    {/* Reset */}
                    <button
                      onClick={resetScanner}
                      className="flex w-full items-center justify-center gap-3 rounded-2xl border border-emerald-300 bg-white px-6 py-4 font-extrabold text-emerald-700 transition hover:bg-emerald-50"
                    >
                      <RotateCcw className="h-5 w-5" />
                      Scan Another Leaf
                    </button>

                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ======================================
          TIPS
      ====================================== */}
      <section
        id="tips"
        className="bg-white px-5 py-16 sm:px-8 lg:py-24"
      >
        <div className="mx-auto max-w-7xl">

          <div className="mb-10 text-center">
            <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-600">
              Better Results
            </p>

            <h3 className="mt-3 text-3xl font-black text-slate-900">
              Tips for accurate scanning
            </h3>
          </div>

          <div className="grid gap-5 md:grid-cols-3">

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50">
                <Sun className="h-6 w-6 text-emerald-600" />
              </div>

              <h4 className="font-extrabold text-slate-900">
                Good lighting
              </h4>

              <p className="mt-2 text-sm leading-6 text-slate-500">
                Capture the leaf in natural or bright, even lighting.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50">
                <Camera className="h-6 w-6 text-emerald-600" />
              </div>

              <h4 className="font-extrabold text-slate-900">
                Clear image
              </h4>

              <p className="mt-2 text-sm leading-6 text-slate-500">
                Keep the leaf in focus and avoid blurry or extremely distant
                photos.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50">
                <Ban className="h-6 w-6 text-emerald-600" />
              </div>

              <h4 className="font-extrabold text-slate-900">
                Avoid distractions
              </h4>

              <p className="mt-2 text-sm leading-6 text-slate-500">
                Try to keep the leaf visible and minimize heavy backgrounds
                or other objects.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* ======================================
          FOOTER
      ====================================== */}
      <footer className="border-t border-slate-200 bg-slate-950 text-white">
        <div className="mx-auto max-w-7xl px-5 py-12 sm:px-8">

          <div className="grid gap-10 md:grid-cols-4">

            <div className="md:col-span-2">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-600">
                  <Leaf className="h-6 w-6 text-white" />
                </div>

                <h4 className="text-2xl font-black">
                  Flora<span className="text-emerald-400">Scan</span>
                </h4>
              </div>

              <p className="mt-5 max-w-md text-sm leading-6 text-slate-400">
                An AI-powered plant disease detection platform designed to
                make plant health analysis simple and accessible.
              </p>

              <div className="mt-6 flex gap-3">
                <a
                  href="#"
                  className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10 text-xs font-black transition hover:bg-emerald-600"
                >
                  GH
                </a>

                <a
                  href="#"
                  className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10 text-xs font-black transition hover:bg-emerald-600"
                >
                  in
                </a>

                <a
                  href="#"
                  className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10 text-xs font-black transition hover:bg-emerald-600"
                >
                  YT
                </a>
              </div>
            </div>

            <div>
              <h5 className="font-extrabold">
                Quick Links
              </h5>

              <div className="mt-4 space-y-3 text-sm text-slate-400">
                <a
                  href="#"
                  className="block transition hover:text-emerald-400"
                >
                  Home
                </a>

                <a
                  href="#scanner"
                  className="block transition hover:text-emerald-400"
                >
                  Scanner
                </a>

                <a
                  href="#tips"
                  className="block transition hover:text-emerald-400"
                >
                  Tips
                </a>
              </div>
            </div>

            <div>
              <h5 className="font-extrabold">
                Credits
              </h5>

              <div className="mt-4 space-y-3 text-sm text-slate-400">
                <p>
                  Founder:{" "}
                  <span className="font-semibold text-white">
                    Srishty Gupta
                  </span>
                </p>

                <p>
                  Programmer:{" "}
                  <span className="font-semibold text-white">
                    Neha Gupta
                  </span>
                </p>
              </div>
            </div>

          </div>

          <div className="mt-10 border-t border-white/10 pt-6 text-center text-xs text-slate-500">
            © {new Date().getFullYear()} FloraScan. All rights reserved.
          </div>

        </div>
      </footer>
    </div>
  );
};

export default DiseaseDetection;