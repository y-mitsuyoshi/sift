#!/usr/bin/env python3
"""
Silent-Face-Anti-Spoofingモデルのダウンロードスクリプト
"""

import os
import requests
import sys
from pathlib import Path

def download_model():
    """
    2.7_80x80_MiniFASNetV2.pthモデルをダウンロードする
    """
    model_dir = Path("app/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "2.7_80x80_MiniFASNetV2.pth"
    
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return
    
    # GitHubのSilent-Face-Anti-SpoofingリポジトリからダウンロードURL
    # 注意: 実際のURLは公式リポジトリで確認してください
    model_url = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    
    print(f"Downloading model from: {model_url}")
    print(f"Saving to: {model_path}")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # 進捗表示
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\nModel downloaded successfully: {model_path}")
        print(f"File size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        print("Please manually download the model from:")
        print("https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
        print(f"And place it at: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
