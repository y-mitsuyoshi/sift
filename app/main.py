from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import cv2
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
    
    # 複数のモデルパスを試行
    possible_paths = [
        "/app/app/models/2.7_80x80_MiniFASNetV2.pth",  # Dockerコンテナ内
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "2.7_80x80_MiniFASNetV2.pth"),  # 相対パス
        "app/models/2.7_80x80_MiniFASNetV2.pth"  # ワーキングディレクトリから
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found model file at: {path}")
            break
    
    try:
        # アクティブチェッカー初期化（MediaPipe）
        active_checker = ActiveChecker()
        logger.info("Active checker initialized successfully")
        
        # パッシブチェッカー初期化（深層学習モデル）
        if model_path:
            try:
                passive_checker = PassiveChecker(model_path, threshold=0.5)  # 閾値を下げる
                logger.info(f"Passive checker initialized successfully with model: {model_path}")
            except Exception as model_error:
                logger.error(f"Failed to load model from {model_path}: {model_error}")
                logger.info("Using improved fallback passive checker")
                passive_checker = ImprovedFallbackPassiveChecker()
        else:
            logger.warning("Model file not found in any of the expected locations:")
            for path in possible_paths:
                logger.warning(f"  - {path}")
            logger.info("Using improved fallback passive checker")
            passive_checker = ImprovedFallbackPassiveChecker()
        
    except Exception as e:
        logger.error(f"Error during checker initialization: {e}", exc_info=True)
        # パッシブチェッカーのみフォールバック、アクティブチェッカーは常にMediaPipe版を使用
        if passive_checker is None:
            passive_checker = ImprovedFallbackPassiveChecker()
        if active_checker is None:
            active_checker = ActiveChecker()  # 常にMediaPipe版を使用
        logger.info("Using MediaPipe active checker and fallback passive checker")

class FallbackPassiveChecker:
    """
    モデルファイルがない場合のフォールバック実装
    基本的な画像解析で判定
    """
    def __init__(self):
        self.threshold = 0.5
        
    def check(self, frames):
        if not frames:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "No frames to analyze"
            }
        
        # 基本的な画像品質チェック
        scores = []
        for frame in frames[:10]:  # 最初の10フレームのみチェック
            try:
                # グレースケール変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # ラプラシアンによるブラー検出
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # 明度の分散（画像の複雑さ）
                brightness_var = gray.var()
                
                # 正規化されたスコア計算
                blur_score = min(laplacian_var / 1000.0, 1.0)
                brightness_score = min(brightness_var / 10000.0, 1.0)
                
                combined_score = (blur_score + brightness_score) / 2.0
                scores.append(combined_score)
                
            except Exception:
                scores.append(0.5)  # デフォルトスコア
        
        average_score = sum(scores) / len(scores) if scores else 0.5
        passed = bool(average_score >= self.threshold)
        
        return {
            "passed": passed,
            "average_real_score": round(float(average_score), 3),
            "message": "Fallback passive check (basic image quality analysis)"
        }

class ImprovedFallbackPassiveChecker:
    """
    改良されたフォールバック実装
    顔検出と画像品質を組み合わせた判定 + 新しい検知器
    """
    def __init__(self):
        self.threshold = 0.3  # より寛容な閾値
        # OpenCVのカスケード分類器を初期化
        try:
            import cv2
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.has_face_detector = True
        except:
            self.has_face_detector = False
            logger.warning("Face cascade not available, using basic image analysis only")
        
        # 新しい検知器をインポート・初期化
        try:
            from .passive_checker import TextureNoiseDetector, FaceIdentityDetector, VirtualCameraDetector
            self.texture_detector = TextureNoiseDetector()
            self.identity_detector = FaceIdentityDetector()
            self.virtual_camera_detector = VirtualCameraDetector()
            self.enhanced_available = True
            logger.info("Enhanced detectors initialized successfully")
        except ImportError as e:
            logger.warning(f"Enhanced detectors not available: {e}")
            self.enhanced_available = False
        
    def check(self, frames):
        if not frames:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "No frames to analyze",
                "enhanced_analysis": None
            }
        
        # 基本的な画像品質分析
        scores = []
        face_detection_count = 0
        
        for frame in frames[:15]:  # より多くのフレームをチェック
            try:
                # グレースケール変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 顔検出スコア
                face_score = 0.0
                if self.has_face_detector:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        face_detection_count += 1
                        # 顔のサイズに基づくスコア
                        largest_face = max(faces, key=lambda x: x[2] * x[3])
                        face_area = largest_face[2] * largest_face[3]
                        total_area = gray.shape[0] * gray.shape[1]
                        face_ratio = face_area / total_area
                        face_score = min(face_ratio * 10, 1.0)  # 顔の大きさに基づく
                
                # 画像品質スコア
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness_var = gray.var()
                
                # エッジ検出（Canny）
                edges = cv2.Canny(gray, 50, 150)
                edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
                
                # 正規化されたスコア計算
                blur_score = min(laplacian_var / 500.0, 1.0)  # より寛容
                brightness_score = min(brightness_var / 5000.0, 1.0)  # より寛容
                edge_score = min(edge_density * 20, 1.0)
                
                # 総合スコア（顔検出があれば重み付け）
                if face_score > 0:
                    combined_score = (face_score * 0.4 + blur_score * 0.2 + brightness_score * 0.2 + edge_score * 0.2)
                else:
                    combined_score = (blur_score * 0.4 + brightness_score * 0.3 + edge_score * 0.3)
                
                scores.append(combined_score)
                
            except Exception as e:
                logger.warning(f"Error processing frame: {e}")
                scores.append(0.3)  # デフォルトスコア
        
        if not scores:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "Failed to process any frames",
                "enhanced_analysis": None
            }
        
        base_average_score = sum(scores) / len(scores)
        face_detection_rate = face_detection_count / len(frames[:15]) if self.has_face_detector else 0
        
        # 顔検出率も考慮
        if face_detection_rate > 0.3:  # 30%以上のフレームで顔検出
            base_average_score += 0.2  # ボーナス
        
        base_average_score = min(base_average_score, 1.0)
        
        # 強化された検知器を実行（利用可能な場合）
        enhanced_analysis = None
        final_score = base_average_score
        
        if self.enhanced_available:
            try:
                logger.info("Running enhanced fallback analysis...")
                
                # 各検知器を実行
                texture_result = self.texture_detector.compute_texture_score(frames)
                identity_result = self.identity_detector.analyze_face_consistency(frames)
                virtual_camera_result = self.virtual_camera_detector.detect_virtual_camera()
                
                # 強化分析の結果を統合
                enhanced_weights = {
                    'base': 0.5,
                    'texture': 0.2,
                    'identity': 0.2,
                    'virtual_camera': 0.1
                }
                
                enhanced_scores = {
                    'base': base_average_score,
                    'texture': texture_result['average_score'],
                    'identity': identity_result['consistency_score'],
                    'virtual_camera': 1.0 - virtual_camera_result['risk_score']
                }
                
                # 加重平均で最終スコア計算
                final_score = sum(
                    enhanced_scores[key] * enhanced_weights[key] 
                    for key in enhanced_weights.keys()
                )
                
                enhanced_analysis = {
                    "texture_analysis": texture_result,
                    "identity_analysis": identity_result,
                    "virtual_camera_analysis": virtual_camera_result,
                    "enhanced_scores": {k: round(v, 3) for k, v in enhanced_scores.items()},
                    "weights_used": enhanced_weights
                }
                
            except Exception as e:
                logger.warning(f"Enhanced analysis failed, falling back to basic analysis: {e}")
                final_score = base_average_score
        
        passed = bool(final_score >= self.threshold)
        
        return {
            "passed": passed,
            "average_real_score": round(float(final_score), 3),
            "base_score": round(float(base_average_score), 3),
            "face_detection_rate": round(face_detection_rate, 3),
            "message": f"{'Enhanced' if enhanced_analysis else 'Basic'} fallback check ({'passed' if passed else 'failed'})",
            "enhanced_analysis": enhanced_analysis
        }

def validate_video_file(file: UploadFile) -> None:
    """
    動画ファイルの検証
    """
    # ファイルサイズチェック（100MB）
    max_size = 100 * 1024 * 1024  # 100MB

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
        
        # 30秒制限チェック
        if video_info["duration"] > 30.0:
            raise HTTPException(
                status_code=400,
                detail=f"Video duration ({video_info['duration']:.2f}s) exceeds 30 second limit"
            )
        
        # フレーム抽出
        frames = extract_frames(temp_file_path, max_frames=450)  # 最大450フレーム（30秒×15fps相当）
        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video")
        
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # パッシブチェック実行
        logger.info("Starting passive check...")
        passive_result = passive_checker.check(frames)
        logger.info(f"Passive check result: {passive_result}")
        
        # アクティブチェック実行（パッシブチェック結果に関わらず実行）
        logger.info("Starting active check...")
        active_result = active_checker.check(frames)
        logger.info(f"Active check result: {active_result}")
        
        # 最終判定：両方のチェックを組み合わせて判定
        # パッシブチェックが低スコアでも、アクティブチェックが成功すれば合格とする
        passive_passed = passive_result["passed"]
        active_passed = active_result["passed"]
        
        # 判定ロジック：
        # 1. 両方成功 → SUCCESS
        # 2. アクティブのみ成功 + パッシブスコアが極端に低くない(>0.1) → SUCCESS  
        # 3. その他 → FAILURE
        passive_score = passive_result.get("average_real_score", 0.0)
        
        if passive_passed and active_passed:
            status = "SUCCESS"
            reason = "Both passive and active checks passed."
        elif active_passed and passive_score > 0.1:
            status = "SUCCESS" 
            reason = "Active check passed and passive check score acceptable."
        elif not active_passed:
            status = "FAILURE"
            reason = f"Active challenge failed: {active_result['message']}"
        else:
            status = "FAILURE"
            reason = f"Passive check failed with low score: {passive_score:.3f}"
        
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
