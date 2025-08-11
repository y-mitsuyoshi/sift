import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Tuple
from collections import deque

from .models.challenges import LivenessAction

class ActiveChecker:
    def __init__(self):
        """
        Initializes the active liveness checker.
        """
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Detection Thresholds ---
        self.BLINK_THRESHOLD = 0.20  # Eye Aspect Ratio threshold for blink
        self.MOUTH_OPEN_THRESHOLD = 0.4  # Mouth Aspect Ratio threshold
        self.HEAD_TURN_YAW_THRESHOLD = 0.3  # Ratio of face width for yaw turn
        self.NOD_THRESHOLD = 0.02  # Normalized vertical movement for nod
        self.NOD_FRAME_BUFFER = 10  # Frames to analyze for a nod

        # State management will be reset for each check
        self.reset_state()

    def reset_state(self):
        """Resets the state for a new liveness check."""
        self.challenge_sequence: List[LivenessAction] = []
        self.current_step_index = 0
        self.challenge_completed = False

        # State for specific actions
        self.blink_count = 0
        self.is_eye_closed = False
        self.closed_frames = 0
        self.last_blink_frame = 0
        self.nose_y_history = deque(maxlen=self.NOD_FRAME_BUFFER)
        self.frame_count = 0

    def _calculate_ear(self, eye_landmarks, face_landmarks, frame_shape):
        """Calculates Eye Aspect Ratio, normalized by face height."""
        p1 = np.array([face_landmarks.landmark[eye_landmarks[0]].x, face_landmarks.landmark[eye_landmarks[0]].y])
        p2 = np.array([face_landmarks.landmark[eye_landmarks[1]].x, face_landmarks.landmark[eye_landmarks[1]].y])
        p3 = np.array([face_landmarks.landmark[eye_landmarks[2]].x, face_landmarks.landmark[eye_landmarks[2]].y])
        p4 = np.array([face_landmarks.landmark[eye_landmarks[3]].x, face_landmarks.landmark[eye_landmarks[3]].y])
        p5 = np.array([face_landmarks.landmark[eye_landmarks[4]].x, face_landmarks.landmark[eye_landmarks[4]].y])
        p6 = np.array([face_landmarks.landmark[eye_landmarks[5]].x, face_landmarks.landmark[eye_landmarks[5]].y])

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal < 1e-6:
            return 0.3 # Default for safety

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _detect_blink(self, landmarks) -> bool:
        """Detects if the user has blinked twice."""
        # Using refined landmarks for eyes
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        left_ear = self._calculate_ear(left_eye_indices, landmarks, None)
        right_ear = self._calculate_ear(right_eye_indices, landmarks, None)
        avg_ear = (left_ear + right_ear) / 2.0

        blink_detected_this_frame = False
        if avg_ear < self.BLINK_THRESHOLD:
            self.closed_frames += 1
            if not self.is_eye_closed and self.closed_frames >= 2:
                self.is_eye_closed = True
        else:
            if self.is_eye_closed and self.frame_count - self.last_blink_frame > 5:
                self.blink_count += 1
                blink_detected_this_frame = True
                self.last_blink_frame = self.frame_count
            self.is_eye_closed = False
            self.closed_frames = 0
        
        return self.blink_count >= 2

    def _detect_head_turn(self, landmarks) -> str:
        """Detects head yaw (turning left or right)."""
        nose_tip = landmarks.landmark[1].x
        left_face_edge = landmarks.landmark[130].x # A point on the left side of the face
        right_face_edge = landmarks.landmark[359].x # A point on the right side of the face
        
        face_width = right_face_edge - left_face_edge
        if face_width < 1e-6: return 'center'
        
        nose_ratio = (nose_tip - left_face_edge) / face_width
        
        if nose_ratio < self.HEAD_TURN_YAW_THRESHOLD:
            return 'left'
        if nose_ratio > (1.0 - self.HEAD_TURN_YAW_THRESHOLD):
            return 'right'
        return 'center'

    def _detect_mouth_open(self, landmarks) -> bool:
        """Detects if the mouth is open."""
        upper_lip_y = landmarks.landmark[13].y
        lower_lip_y = landmarks.landmark[14].y
        face_top_y = landmarks.landmark[10].y
        face_bottom_y = landmarks.landmark[152].y
        
        face_height = face_bottom_y - face_top_y
        if face_height < 1e-6: return False
        
        mouth_height = abs(lower_lip_y - upper_lip_y)
        mouth_ratio = mouth_height / face_height
        
        return mouth_ratio > self.MOUTH_OPEN_THRESHOLD

    def _detect_nod(self, landmarks) -> bool:
        """Detects a nodding motion."""
        nose_tip_y = landmarks.landmark[1].y
        self.nose_y_history.append(nose_tip_y)
        
        if len(self.nose_y_history) < self.NOD_FRAME_BUFFER:
            return False

        face_top_y = landmarks.landmark[10].y
        face_bottom_y = landmarks.landmark[152].y
        face_height = face_bottom_y - face_top_y
        if face_height < 1e-6: return False
            
        vertical_movement = (max(self.nose_y_history) - min(self.nose_y_history)) / face_height

        if vertical_movement > self.NOD_THRESHOLD:
            self.nose_y_history.clear() # Reset after detection
            return True
        return False

    def _process_frame(self, frame: np.ndarray):
        """Processes a single frame to check for the current challenge."""
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return
        
        # We only process the first detected face
        face_landmarks = results.multi_face_landmarks[0]

        if self.challenge_completed:
            return

        current_challenge = self.challenge_sequence[self.current_step_index]
        action_detected = False

        detector_map = {
            LivenessAction.BLINK_TWICE: lambda lm: self._detect_blink(lm),
            LivenessAction.TURN_LEFT: lambda lm: self._detect_head_turn(lm) == 'left',
            LivenessAction.TURN_RIGHT: lambda lm: self._detect_head_turn(lm) == 'right',
            LivenessAction.OPEN_MOUTH: lambda lm: self._detect_mouth_open(lm),
            LivenessAction.NOD: lambda lm: self._detect_nod(lm),
        }
        
        detector = detector_map.get(current_challenge)
        if detector and detector(face_landmarks):
            action_detected = True

        if action_detected:
            self.current_step_index += 1
            # Reset states for single-action detectors to prevent immediate re-triggering
            if current_challenge in [LivenessAction.NOD, LivenessAction.OPEN_MOUTH]:
                # A short delay or state clear might be needed, handled by check loop
                pass
            
            if self.current_step_index >= len(self.challenge_sequence):
                self.challenge_completed = True

    def check(self, frames: List[np.ndarray], challenge_sequence: List[LivenessAction]) -> Dict[str, any]:
        """
        Checks a sequence of frames against a dynamic challenge sequence.
        """
        self.reset_state()
        self.challenge_sequence = challenge_sequence

        if not frames:
            return {"passed": False, "message": "No frames to analyze", "details": {}}
        
        for frame in frames:
            if self.challenge_completed:
                break
            try:
                self._process_frame(frame)
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing frame {self.frame_count}: {e}")
                continue

        passed = self.challenge_completed
        message = ""
        if passed:
            message = "Challenge sequence completed successfully."
        else:
            if self.current_step_index < len(self.challenge_sequence):
                failed_challenge = self.challenge_sequence[self.current_step_index]
                message = f"Challenge failed at step {self.current_step_index + 1}: Did not detect '{failed_challenge.value}'."
            else:
                message = "Challenge not completed."

        return {
            "passed": passed,
            "message": message,
            "details": {
                "challenges_requested": [c.name for c in self.challenge_sequence],
                "challenges_completed": self.current_step_index,
                "total_frames_processed": self.frame_count,
            }
        }
