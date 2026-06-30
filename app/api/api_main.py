"""
Support Ticket Priority Classifier - FastAPI Backend
Serves the Bi-LSTM model and handles real-time predictions.
"""

import os
import pickle
import logging
from typing import Dict, List
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (update these to match your repo structure)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(BASE_DIR / "models" / "ticket_classifier.keras")
TOKENIZER_PATH = str(BASE_DIR / "data" / "processed" / "vocab.pkl")

# Model hyperparameters (must match training)
MAX_SEQ_LEN = 57
NUM_CLASSES = 3
VOCAB_SIZE = 5726

# Label mapping
LABEL_TO_ID = {"Low": 0, "Medium": 1, "High": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Department routing (from Ticket Type)
DEPARTMENT_MAPPING = {
    "Technical issue": "Technical Support",
    "Billing inquiry": "Billing",
    "Refund request": "Billing",
    "Cancellation request": "Customer Retention",
    "Product inquiry": "Sales"
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE (loaded at startup)
# ============================================================================

class AppState:
    """Holds model and tokenizer for the lifetime of the app."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

app_state = AppState()

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model and tokenizer on startup.
    Clean up on shutdown.
    """
    # STARTUP
    logger.info("🚀 Initializing Support Ticket Priority Classifier API...")
    
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        app_state.model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
        
        # Load tokenizer
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
        
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, 'rb') as f:
            vocab = pickle.load(f)
            
        app_state.tokenizer = TextVectorization(output_mode='int')
        app_state.tokenizer.set_vocabulary(vocab)
        logger.info("✅ Tokenizer loaded successfully")
        
        app_state.model_loaded = True
        logger.info("✅ API is ready to serve predictions!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model/tokenizer: {str(e)}")
        logger.error("Make sure to run: python src/data/download_data.py && python src/models/train_classifier.py")
        app_state.model_loaded = False
    
    yield
    
    # SHUTDOWN
    logger.info("🛑 Shutting down API...")
    if app_state.model is not None:
        del app_state.model
    if app_state.tokenizer is not None:
        del app_state.tokenizer
    logger.info("✅ Cleanup complete")

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Support Ticket Priority Classifier",
    description="Real-time ticket classification using Bi-LSTM deep learning model",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (allow requests from Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class PredictRequest(BaseModel):
    """Request payload for ticket classification."""
    subject: str = Field(..., min_length=1, max_length=200, description="Ticket subject line")
    body: str = Field(..., min_length=1, max_length=3000, description="Ticket description")
    ticket_type: str = Field(
        default="Technical issue",
        description="Type of ticket (for department routing)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Cannot login to account",
                "body": "I've been trying to log in all morning but getting an error...",
                "ticket_type": "Technical issue"
            }
        }


class PredictResponse(BaseModel):
    """Response payload with classification results."""
    priority: str = Field(..., description="Predicted priority level")
    confidence: Dict[str, float] = Field(..., description="Confidence scores for each priority class")
    department: str = Field(..., description="Recommended department for routing")
    model_version: str = Field(default="1.0", description="Model version")

    class Config:
        json_schema_extra = {
            "example": {
                "priority": "High",
                "confidence": {
                    "Low": 0.05,
                    "Medium": 0.10,
                    "High": 0.70,
                    "Critical": 0.15
                },
                "department": "Technical Support",
                "model_version": "1.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is ready")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess ticket text to match training pipeline.
    
    This should mirror preprocessing.py logic.
    For now, simple lowercasing + whitespace cleanup.
    """
    text = text.lower()
    text = " ".join(text.split())  # Normalize whitespace
    return text


def tokenize_and_pad(text: str) -> np.ndarray:
    """
    Tokenize text and pad to MAX_SEQ_LEN.
    
    Args:
        text: Input text
        
    Returns:
        Padded sequence array of shape (1, MAX_SEQ_LEN)
    """
    # Tokenize
    sequences = app_state.tokenizer([text])
    
    # Pad to MAX_SEQ_LEN
    padded = pad_sequences(sequences.numpy(), maxlen=MAX_SEQ_LEN, padding='pre', truncating='post')
    
    return padded


def run_inference(padded_sequence: np.ndarray) -> np.ndarray:
    """
    Run model inference on padded sequence.
    
    Args:
        padded_sequence: Padded token sequence of shape (1, MAX_SEQ_LEN)
        
    Returns:
        Confidence scores of shape (1, NUM_CLASSES)
    """
    predictions = app_state.model.predict(padded_sequence, verbose=0)
    return predictions


def format_confidence(predictions: np.ndarray) -> Dict[str, float]:
    """
    Convert model predictions to confidence dictionary.
    
    Args:
        predictions: Raw model output of shape (1, NUM_CLASSES)
        
    Returns:
        Dictionary mapping priority label to confidence score
    """
    scores = predictions[0]  # Get first (only) sample
    confidence_dict = {
        ID_TO_LABEL[i]: float(scores[i])
        for i in range(NUM_CLASSES)
    }
    return confidence_dict


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and model status.
    """
    return HealthResponse(
        status="healthy" if app_state.model_loaded else "unhealthy",
        model_loaded=app_state.model_loaded
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify a support ticket in real-time.
    
    Takes ticket subject + body and returns:
    - Priority level (Low, Medium, High, Critical)
    - Confidence scores for each priority
    - Recommended department for routing
    
    Args:
        request: Ticket classification request
        
    Returns:
        Classification result with predictions
    """
    # Check if model is loaded
    if not app_state.model_loaded:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API is not ready. Check server logs."
        )
    
    try:
        # Combine subject and body for classification
        combined_text = f"{request.subject} {request.body}"
        
        # Preprocess
        preprocessed = preprocess_text(combined_text)
        logger.debug(f"Preprocessed text: {preprocessed[:100]}...")
        
        # Tokenize and pad
        padded = tokenize_and_pad(preprocessed)
        logger.debug(f"Padded shape: {padded.shape}")
        
        # Run inference
        predictions = run_inference(padded)
        logger.debug(f"Predictions: {predictions}")
        
        # Get predicted priority
        priority_id = np.argmax(predictions[0])
        priority_label = ID_TO_LABEL[priority_id]
        
        # Format confidence scores
        confidence = format_confidence(predictions)
        
        # Route to department based on ticket_type
        department = DEPARTMENT_MAPPING.get(request.ticket_type, "General Support")
        
        logger.info(
            f"✅ Classified ticket: {request.subject[:50]}... → {priority_label} "
            f"({confidence[priority_label]:.2%}) → {department}"
        )
        
        return PredictResponse(
            priority=priority_label,
            confidence=confidence,
            department=department,
            model_version="1.0"
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Support Ticket Priority Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        },
        "model_loaded": app_state.model_loaded
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run with: python app/api/main.py
    # Or: uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
    
    logger.info("Starting Support Ticket Classifier API...")
    logger.info("Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
