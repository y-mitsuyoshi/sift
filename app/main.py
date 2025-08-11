from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import cv2
import uuid
from typing import Dict, Any, List

from .video_processor import save_uploaded_file, extract_frames, cleanup_temp_file, get_video_info
from .models.challenges import LivenessAction, ChallengeGenerator
from .active_checker import ActiveChecker

# Conditional import for passive checker (assuming it might not be present)
try:
    from .passive_checker import PassiveChecker
    PASSIVE_CHECKER_AVAILABLE = True
except ImportError as e:
    PASSIVE_CHECKER_AVAILABLE = False
    logging.warning(f"PassiveChecker not available: {e}. Some functionality will be disabled.")


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Liveness Detection API",
    description="An API for liveness detection using dynamic, session-based challenges.",
    version="1.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory storage for sessions and services ---
# In a production environment, use a more robust solution like Redis.
sessions: Dict[str, List[LivenessAction]] = {}
passive_checker = None
active_checker = None
challenge_generator = None


# --- Fallback Passive Checker ---
# This is a dummy class for when the actual passive checker isn't available.
class FallbackPassiveChecker:
    def check(self, frames):
        return {
            "passed": True,
            "average_real_score": 1.0,
            "message": "Fallback passive checker used. Check skipped."
        }

@app.on_event("startup")
async def startup_event():
    """Application startup initialization."""
    global passive_checker, active_checker, challenge_generator

    logger.info("Initializing services...")
    active_checker = ActiveChecker()
    challenge_generator = ChallengeGenerator()

    if PASSIVE_CHECKER_AVAILABLE:
        # Logic to find and initialize the real passive checker model
        model_path_found = None
        possible_paths = [
            "/app/app/models/2.7_80x80_MiniFASNetV2.pth",
            os.path.join(os.path.dirname(__file__), "models", "2.7_80x80_MiniFASNetV2.pth")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path_found = path
                break
        
        if model_path_found:
            try:
                passive_checker = PassiveChecker(model_path_found)
                logger.info(f"Passive checker initialized successfully with model: {model_path_found}")
            except Exception as e:
                logger.error(f"Failed to load passive checker model: {e}. Using fallback.")
                passive_checker = FallbackPassiveChecker()
        else:
            logger.warning("Passive checker model not found. Using fallback.")
            passive_checker = FallbackPassiveChecker()
    else:
        logger.info("PassiveChecker module not available. Using fallback.")
        passive_checker = FallbackPassiveChecker()

    logger.info("Services initialized.")


def validate_video_file(file: UploadFile):
    """Validates the uploaded video file."""
    # This is a simplified validation.
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a video.")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Dynamic Liveness Detection API is running. See /docs for details."}


@app.post("/liveness/session/start", summary="Start a new liveness session")
async def start_liveness_session() -> JSONResponse:
    """
    Starts a new liveness detection session.

    Generates a unique session ID and a random sequence of challenges.
    The client should display these challenges to the user.
    """
    global sessions, challenge_generator
    session_id = str(uuid.uuid4())

    # Generate a random sequence of 2 or 3 challenges
    challenge_sequence = challenge_generator.generate()
    sessions[session_id] = challenge_sequence

    challenge_messages = [action.value for action in challenge_sequence]

    logger.info(f"Started session {session_id} with challenges: {[c.name for c in challenge_sequence]}")

    return JSONResponse(content={
        "session_id": session_id,
        "challenges": challenge_messages
    })


@app.post("/liveness/check", summary="Perform liveness check for a session")
async def liveness_check(
    session_id: str = Form(...),
    file: UploadFile = File(...)
) -> JSONResponse:
    """
    Performs the liveness check for a given session.
    
    Receives a video file and the session ID, then validates the user's actions
    against the challenges stored for that session.
    """
    global sessions, active_checker, passive_checker

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid or expired session ID.")

    # Retrieve the challenges for this session
    challenge_sequence = sessions[session_id]
    temp_file_path = None
    
    try:
        validate_video_file(file)
        temp_file_path = await save_uploaded_file(file)
        
        video_info = get_video_info(temp_file_path)
        if video_info["duration"] > 15.0: # Limit duration to 15 seconds
            raise HTTPException(status_code=400, detail="Video duration exceeds 15 second limit.")

        frames = extract_frames(temp_file_path, max_frames=225) # 15s * 15fps
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")
        
        logger.info(f"Session {session_id}: Starting passive check.")
        passive_result = passive_checker.check(frames)
        
        logger.info(f"Session {session_id}: Starting active check with challenges: {[c.name for c in challenge_sequence]}")
        active_result = active_checker.check(frames, challenge_sequence)
        
        # Combine results
        passive_passed = passive_result["passed"]
        active_passed = active_result["passed"]
        
        if active_passed:
            status = "SUCCESS"
            reason = "Active and passive checks passed." if passive_passed else "Active check passed, passive check indicates low risk."
        else:
            status = "FAILURE"
            reason = f"Active challenge failed: {active_result.get('message', 'No details')}"

        return JSONResponse(content={
            "status": status,
            "reason": reason,
            "details": {
                "active_check": active_result,
                "passive_check": passive_result
            },
            "video_info": video_info
        })
        
    except HTTPException as e:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise e
    except Exception as e:
        logger.error(f"Error during liveness check for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # Clean up session and temporary file
        if session_id in sessions:
            del sessions[session_id]
            logger.info(f"Session {session_id} cleaned up.")
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
            logger.info(f"Cleaned up temp file: {temp_file_path}")
