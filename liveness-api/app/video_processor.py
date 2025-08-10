import cv2
import numpy as np
import tempfile
import os
from typing import List, Optional
from fastapi import UploadFile

async def save_uploaded_file(upload_file: UploadFile) -> str:
    """
    アップロードされたファイルを一時ファイルとして保存し、パスを返す
    """
    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await upload_file.read()
        temp_file.write(content)
        return temp_file.name

def extract_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    動画ファイルからフレームを抽出し、Numpy配列のリストとして返す
    
    Args:
        video_path: 動画ファイルのパス
        max_frames: 最大フレーム数（Noneの場合は全フレーム）
    
    Returns:
        フレームのリスト
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        # 最大フレーム数の制限
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return frames

def cleanup_temp_file(file_path: str) -> None:
    """
    一時ファイルを削除
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temp file {file_path}: {e}")

def get_video_info(video_path: str) -> dict:
    """
    動画ファイルの情報を取得
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    }
