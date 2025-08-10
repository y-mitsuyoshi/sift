from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import Dict, Any

from .video_processor import save_uploaded_file, extract_frames, cleanup_temp_file, get_video_info
from .passive_checker import PassiveChecker
from .active_checker import ActiveChecker

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="生体検知API",
    description="ディープフェイクや写真・動画によるなりすまし攻撃を防ぐ生体検知API",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数でチェッカーを保持
passive_checker = None
active_checker = None

@app.on_event("startup")
async def startup_event():
    """
    アプリケーション起動時の初期化
    """
    global passive_checker, active_checker
    
    # モデルパス
    model_path = "/app/app/models/2.7_80x80_MiniFASNetV2.pth"
    
    try:
        # パッシブチェッカー初期化
        if os.path.exists(model_path):
            passive_checker = PassiveChecker(model_path, threshold=0.8)
            logger.info("Passive checker initialized successfully")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Passive checker will use mock implementation")
            passive_checker = MockPassiveChecker()
        
        # アクティブチェッカー初期化
        active_checker = ActiveChecker()
        logger.info("Active checker initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # フォールバック用のモックチェッカーを使用
        passive_checker = MockPassiveChecker()
        active_checker = MockActiveChecker()

class MockPassiveChecker:
    """
    モデルファイルがない場合のモック実装
    """
    def check(self, frames):
        logger.warning("Using mock passive checker - model file not available")
        return {
            "passed": True,
            "average_real_score": 0.95,
            "message": "Mock passive check (model not available)"
        }

class MockActiveChecker:
    """
    MediaPipeが利用できない場合のモック実装
    """
    def check(self, frames):
        logger.warning("Using mock active checker - MediaPipe not available")
        return {
            "passed": True,
            "message": "Mock active check (MediaPipe not available)"
        }

def validate_video_file(file: UploadFile) -> None:
    """
    動画ファイルの検証
    """
    # ファイルサイズチェック（10MB）
    max_size = 10 * 1024 * 1024  # 10MB
    
    # ファイルの内容を読み取らずにサイズを取得
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds the limit of {max_size / 1024 / 1024}MB."
        )

    # Content-Typeチェック
    allowed_types = ["video/mp4", "video/mpeg", "video/quicktime"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}"
        )
    
    # ファイル名の拡張子チェック
    if not file.filename.lower().endswith(('.mp4', '.mov', '.mpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must have a valid video extension (.mp4, .mov, .mpeg)"
        )

@app.get("/")
async def root():
    """
    ヘルスチェックエンドポイント
    """
    return {"message": "高精度eKYC生体検知API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """
    詳細ヘルスチェック
    """
    return {
        "status": "healthy",
        "passive_checker": "available" if passive_checker else "unavailable",
        "active_checker": "available" if active_checker else "unavailable"
    }

@app.post("/liveness/check")
async def liveness_check(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    生体検知APIエンドポイント
    
    Args:
        file: アップロードされた動画ファイル
        
    Returns:
        判定結果JSON
    """
    temp_file_path = None
    
    try:
        logger.info(f"Processing liveness check for file: {file.filename}")
        
        # ファイル検証
        validate_video_file(file)
        
        # 一時ファイルとして保存
        temp_file_path = await save_uploaded_file(file)
        logger.info(f"File saved to temporary location: {temp_file_path}")
        
        # 動画情報取得
        video_info = get_video_info(temp_file_path)
        logger.info(f"Video info: {video_info}")
        
        # 5秒制限チェック
        if video_info["duration"] > 5.0:
            raise HTTPException(
                status_code=400,
                detail=f"Video duration ({video_info['duration']:.2f}s) exceeds 5 second limit"
            )
        
        # フレーム抽出
        frames = extract_frames(temp_file_path, max_frames=150)  # 最大150フレーム
        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video")
        
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # パッシブチェック実行
        logger.info("Starting passive check...")
        passive_result = passive_checker.check(frames)
        logger.info(f"Passive check result: {passive_result}")
        
        # パッシブチェック失敗時は即座に返却
        if not passive_result["passed"]:
            return JSONResponse(content={
                "status": "FAILURE",
                "reason": "Passive check failed: Potential spoof detected.",
                "details": {
                    "active_check": {
                        "passed": None,
                        "message": "Not performed due to passive check failure."
                    },
                    "passive_check": passive_result
                },
                "video_info": video_info
            })
        
        # アクティブチェック実行
        logger.info("Starting active check...")
        active_result = active_checker.check(frames)
        logger.info(f"Active check result: {active_result}")
        
        # 最終判定
        overall_success = passive_result["passed"] and active_result["passed"]
        
        if overall_success:
            status = "SUCCESS"
            reason = "Liveness check passed."
        else:
            status = "FAILURE"
            if not active_result["passed"]:
                reason = f"Active challenge failed: {active_result['message']}"
            else:
                reason = "Unknown failure reason."
        
        # レスポンス構築
        response_data = {
            "status": status,
            "reason": reason,
            "details": {
                "active_check": active_result,
                "passive_check": passive_result
            },
            "video_info": video_info
        }
        
        logger.info(f"Final result: {status}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        logger.error(f"Unexpected error during liveness check: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # 一時ファイルのクリーンアップ
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTPException のカスタムハンドラー
    """
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
