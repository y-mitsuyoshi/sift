#!/usr/bin/env python3
"""
APIテスト用スクリプト
"""

import requests
import sys
from pathlib import Path

def test_api(video_path: str, api_url: str = "http://localhost:8000"):
    """
    APIをテストする
    
    Args:
        video_path: テスト用動画ファイルのパス
        api_url: APIのベースURL
    """
    endpoint = f"{api_url}/liveness/check"
    
    # ファイルの存在確認
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Testing API with video: {video_path}")
    print(f"Endpoint: {endpoint}")
    
    try:
        # ファイルをPOST
        with open(video_path, 'rb') as f:
            files = {'file': (Path(video_path).name, f, 'video/mp4')}
            
            print("Sending request...")
            response = requests.post(endpoint, files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(response.json())
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_health_check(api_url: str = "http://localhost:8000"):
    """
    ヘルスチェックをテストする
    """
    try:
        response = requests.get(f"{api_url}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <video_file_path> [api_url]")
        print("Example: python test_api.py test_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    # ヘルスチェック
    print("=== Health Check ===")
    test_health_check(api_url)
    print()
    
    # API テスト
    print("=== API Test ===")
    test_api(video_path, api_url)
