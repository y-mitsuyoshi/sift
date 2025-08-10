import cv2
import numpy as np
from typing import List

def extract_frames(video_path: str) -> List[np.ndarray]:
    """
    動画ファイルからフレームを抽出し、Numpy配列のリストとして返す
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
