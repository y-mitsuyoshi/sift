import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import numpy as np

# Add the project root to the path to allow imports from 'app'
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app, sessions
from app.active_checker import ActiveChecker
from app.models.challenges import LivenessAction

# --- Fixtures ---

@pytest.fixture
def client():
    """Provides a TestClient instance for API testing, ensuring startup events are run."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Ensures the sessions dictionary is clear before and after each test."""
    sessions.clear()
    yield
    sessions.clear()

@pytest.fixture
def checker():
    """Provides an ActiveChecker instance for unit testing."""
    return ActiveChecker()

# --- Unit Tests for ActiveChecker ---

@pytest.mark.parametrize("sequence", [
    [LivenessAction.NOD, LivenessAction.OPEN_MOUTH],
    [LivenessAction.TURN_LEFT, LivenessAction.BLINK_TWICE],
    [LivenessAction.TURN_RIGHT, LivenessAction.NOD, LivenessAction.OPEN_MOUTH]
])
def test_active_checker_success_sequences(checker, sequence):
    """
    Tests that the ActiveChecker's state machine correctly validates various successful sequences.
    This test mocks _process_frame to simulate a user performing actions correctly over time.
    """
    # Arrange
    num_frames = len(sequence)
    mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]

    # This mock simulates that for each frame processed, one step of the challenge is completed.
    def mock_process_frame_success(frame):
        checker.current_step_index += 1
        if checker.current_step_index >= len(checker.challenge_sequence):
            checker.challenge_completed = True

    with patch.object(checker, '_process_frame', side_effect=mock_process_frame_success):
        # Act
        result = checker.check(mock_frames, sequence)

        # Assert
        assert result["passed"] is True
        assert "completed successfully" in result["message"]
        assert result["details"]["challenges_completed"] == len(sequence)

def test_active_checker_failure_wrong_action(checker):
    """
    Tests that ActiveChecker fails if the user performs the wrong action.
    This test simulates a user doing nothing.
    """
    # Arrange
    challenge_sequence = [LivenessAction.TURN_LEFT, LivenessAction.OPEN_MOUTH]

    # Mock _process_frame to do nothing, simulating a non-compliant user.
    with patch.object(checker, '_process_frame', return_value=None):
        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]

        # Act
        result = checker.check(mock_frames, challenge_sequence)

        # Assert
        assert result["passed"] is False
        assert "Challenge failed at step 1" in result["message"]
        # Check for the Japanese message defined in the Enum
        assert LivenessAction.TURN_LEFT.value in result["message"]
        assert result["details"]["challenges_completed"] == 0

# --- API Integration Tests ---

def test_start_session_endpoint(client):
    """
    Tests the /liveness/session/start endpoint.
    """
    # Act
    response = client.post("/liveness/session/start")
    data = response.json()

    # Assert
    assert response.status_code == 200
    assert "session_id" in data
    assert "challenges" in data
    assert isinstance(data["session_id"], str)
    assert isinstance(data["challenges"], list)
    assert len(data["challenges"]) in [2, 3]
    # Verify the session was created on the server
    assert data["session_id"] in sessions
    assert len(sessions[data["session_id"]]) == len(data["challenges"])

@patch("app.main.save_uploaded_file", return_value="dummy_path.mp4")
@patch("app.main.extract_frames", return_value=[np.zeros((100, 100, 3))])
@patch("app.main.get_video_info", return_value={"duration": 5.0, "frame_count": 150, "fps": 30})
@patch("app.main.active_checker.check")
@patch("app.main.passive_checker.check")
def test_liveness_check_e2e_success(mock_passive_check, mock_active_check, mock_get_info, mock_extract, mock_save, client):
    """
    Tests the end-to-end flow for a successful liveness check.
    """
    # Arrange
    mock_active_check.return_value = {"passed": True, "message": "Success"}
    mock_passive_check.return_value = {"passed": True}

    start_response = client.post("/liveness/session/start")
    session_id = start_response.json()["session_id"]
    challenges_from_session = sessions[session_id]

    video_file = ("test.mp4", io.BytesIO(b"dummy video"), "video/mp4")

    # Act
    check_response = client.post(
        "/liveness/check",
        data={"session_id": session_id},
        files={"file": video_file}
    )
    data = check_response.json()

    # Assert
    assert check_response.status_code == 200
    assert data["status"] == "SUCCESS"

    mock_active_check.assert_called_once()
    call_args = mock_active_check.call_args[0]
    assert call_args[1] == challenges_from_session

    assert session_id not in sessions

def test_liveness_check_invalid_session(client):
    """
    Tests that the /liveness/check endpoint returns a 404 for an invalid session ID.
    """
    # Arrange
    video_file = ("test.mp4", io.BytesIO(b"dummy video"), "video/mp4")

    # Act
    response = client.post(
        "/liveness/check",
        data={"session_id": "this-is-not-a-valid-id"},
        files={"file": video_file}
    )

    # Assert
    assert response.status_code == 404
    assert "Invalid or expired session ID" in response.json()["detail"]
