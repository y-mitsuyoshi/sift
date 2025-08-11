import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import List, Dict, Tuple, Optional
import os
import logging
from skimage import feature
from skimage.filters import gaussian
import psutil
import json

# DeepFaceの遅延インポート（初期化時のエラーを防ぐ）
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Face identity analysis will be disabled.")

logger = logging.getLogger(__name__)

class MiniFASNetV2(nn.Module):
    """
    MiniFASNetV2 モデルの定義
    """
    def __init__(self, num_classes=3, conv6_kernel=(5, 5), dropout=0.0):
        super(MiniFASNetV2, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=conv6_kernel, stride=1, padding=0)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn6(self.conv6(x)))
        
        # Global average pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x

class TextureNoiseDetector:
    """
    検知器1: テクスチャ・ノイズ分析
    ディープフェイク特有のデジタル加工痕や不自然な質感を検出
    """
    def __init__(self):
        self.threshold = 0.6
        
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        フレームのテクスチャとノイズを分析
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. LBP (Local Binary Pattern) による質感分析
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_var = np.var(lbp)
        
        # 2. ハイフリケンシーノイズ分析
        high_freq = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        noise_level = np.std(high_freq)
        
        # 3. 圧縮アーティファクト検出
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        compression_artifacts = np.mean(diff)
        
        # 4. エッジ一貫性分析
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. 周波数ドメイン分析
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        freq_anomaly = np.std(magnitude_spectrum)
        
        return {
            'lbp_variance': float(lbp_var),
            'noise_level': float(noise_level),
            'compression_artifacts': float(compression_artifacts),
            'edge_density': float(edge_density),
            'frequency_anomaly': float(freq_anomaly)
        }
    
    def compute_texture_score(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        複数フレームからテクスチャスコアを計算
        """
        scores = []
        detailed_metrics = []
        
        for frame in frames[:10]:  # 最初の10フレームを分析
            metrics = self.analyze_frame(frame)
            detailed_metrics.append(metrics)
            
            # 正常な質感スコア計算（値が高いほど自然）
            texture_score = (
                min(metrics['lbp_variance'] / 50.0, 1.0) * 0.3 +  # LBP多様性
                min(metrics['edge_density'] * 10, 1.0) * 0.2 +     # エッジ密度
                (1.0 - min(metrics['compression_artifacts'] / 10.0, 1.0)) * 0.2 +  # 圧縮少ない
                min(metrics['noise_level'] / 20.0, 1.0) * 0.15 +   # 適度なノイズ
                (1.0 - min(metrics['frequency_anomaly'] / 100.0, 1.0)) * 0.15  # 周波数異常少ない
            )
            
            scores.append(texture_score)
        
        avg_score = np.mean(scores) if scores else 0.0
        passed = avg_score >= self.threshold
        
        return {
            'passed': bool(passed),
            'average_score': round(float(avg_score), 3),
            'individual_scores': [round(float(s), 3) for s in scores],
            'detailed_metrics': detailed_metrics,
            'message': f"Texture analysis {'passed' if passed else 'failed'} (score: {avg_score:.3f})"
        }

class FaceIdentityDetector:
    """
    検知器2: 時間軸上の同一性分析
    フレーム間の顔の人物ID一貫性を検証
    """
    def __init__(self):
        self.threshold = 0.8
        self.available = DEEPFACE_AVAILABLE
        
    def analyze_face_consistency(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        フレーム間の顔一貫性を分析
        """
        if not self.available:
            return {
                'passed': True,  # DeepFaceが利用できない場合はスキップ
                'consistency_score': 1.0,
                'message': "DeepFace not available, skipping identity analysis",
                'frame_similarities': []
            }
        
        try:
            # フレームサンプリング（5-10フレーム）
            sample_indices = np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int)
            sample_frames = [frames[i] for i in sample_indices]
            
            embeddings = []
            valid_frames = []
            
            # 各フレームから顔の特徴ベクトルを抽出
            for i, frame in enumerate(sample_frames):
                try:
                    # フレームを一時ファイルとして保存
                    temp_path = f"/tmp/temp_frame_{i}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # DeepFaceで顔の特徴抽出
                    result = DeepFace.represent(temp_path, model_name='VGG-Face', enforce_detection=False)
                    if result and len(result) > 0:
                        embeddings.append(np.array(result[0]['embedding']))
                        valid_frames.append(i)
                    
                    # 一時ファイル削除
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to process frame {i}: {e}")
                    continue
            
            if len(embeddings) < 2:
                return {
                    'passed': False,
                    'consistency_score': 0.0,
                    'message': "Insufficient valid faces detected",
                    'frame_similarities': []
                }
            
            # フレーム間の類似度計算
            similarities = []
            for i in range(len(embeddings) - 1):
                for j in range(i + 1, len(embeddings)):
                    # コサイン類似度計算
                    cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(float(cos_sim))
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            min_similarity = np.min(similarities) if similarities else 0.0
            
            # 判定: 平均類似度が高く、最小類似度も閾値以上
            passed = avg_similarity >= self.threshold and min_similarity >= (self.threshold - 0.1)
            
            return {
                'passed': bool(passed),
                'consistency_score': round(float(avg_similarity), 3),
                'min_similarity': round(float(min_similarity), 3),
                'frame_similarities': [round(float(s), 3) for s in similarities],
                'valid_frame_count': len(embeddings),
                'message': f"Identity consistency {'passed' if passed else 'failed'} (avg: {avg_similarity:.3f})"
            }
            
        except Exception as e:
            logger.error(f"Face identity analysis failed: {e}")
            return {
                'passed': True,  # エラー時はスキップ
                'consistency_score': 1.0,
                'message': f"Identity analysis error: {str(e)}",
                'frame_similarities': []
            }

class VirtualCameraDetector:
    """
    検知器3: 入力ソースの信頼性分析
    仮想カメラの使用を検出
    """
    def __init__(self):
        self.suspicious_processes = [
            'obs', 'v4l2loopback', 'manycam', 'xsplit', 'snapcamera',
            'facerig', 'avatarify', 'deepfacelive', 'faceit'
        ]
        
    def detect_virtual_camera(self) -> Dict[str, any]:
        """
        仮想カメラの使用を検出
        """
        try:
            # 実行中のプロセスをチェック
            suspicious_found = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower() if proc_info['name'] else ''
                    cmdline = ' '.join(proc_info['cmdline']).lower() if proc_info['cmdline'] else ''
                    
                    for suspicious in self.suspicious_processes:
                        if suspicious in proc_name or suspicious in cmdline:
                            suspicious_found.append({
                                'process': proc_info['name'],
                                'pid': proc_info['pid'],
                                'cmdline': cmdline[:100]  # 最初の100文字のみ
                            })
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Linux特有の仮想カメラチェック
            virtual_devices = []
            if os.path.exists('/dev'):
                try:
                    for device in os.listdir('/dev'):
                        if device.startswith('video'):
                            device_path = f'/dev/{device}'
                            try:
                                # v4l2-ctlでデバイス情報を取得（可能な場合）
                                import subprocess
                                result = subprocess.run(
                                    ['v4l2-ctl', '--device', device_path, '--info'],
                                    capture_output=True, text=True, timeout=2
                                )
                                if 'loopback' in result.stdout.lower():
                                    virtual_devices.append(device_path)
                            except (subprocess.TimeoutExpired, FileNotFoundError):
                                pass
                except PermissionError:
                    pass
            
            # リスク評価
            risk_score = 0.0
            if suspicious_found:
                risk_score += 0.7  # 疑わしいプロセス発見
            if virtual_devices:
                risk_score += 0.5  # 仮想デバイス発見
            
            risk_score = min(risk_score, 1.0)
            passed = risk_score < 0.5
            
            return {
                'passed': bool(passed),
                'risk_score': round(float(risk_score), 3),
                'suspicious_processes': suspicious_found,
                'virtual_devices': virtual_devices,
                'message': f"Virtual camera detection {'passed' if passed else 'failed'} (risk: {risk_score:.3f})"
            }
            
        except Exception as e:
            logger.error(f"Virtual camera detection failed: {e}")
            return {
                'passed': True,  # エラー時は通す
                'risk_score': 0.0,
                'message': f"Virtual camera detection error: {str(e)}",
                'suspicious_processes': [],
                'virtual_devices': []
            }

class PassiveChecker:
    def __init__(self, model_path: str, threshold: float = 0.8):
        """
        パッシブ検知器の初期化
        
        Args:
            model_path: 学習済みモデルのパス
            threshold: 判定閾値
        """
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_size = (80, 80)
        
        # 新しい検知器を初期化
        self.texture_detector = TextureNoiseDetector()
        self.identity_detector = FaceIdentityDetector()
        self.virtual_camera_detector = VirtualCameraDetector()
        
        self._load_model()

    def _load_model(self):
        """
        学習済みモデルをロード
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # モデルの初期化
        self.model = MiniFASNetV2(num_classes=3, conv6_kernel=(5, 5))
        
        # 学習済み重みをロード
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # チェックポイントから状態辞書を取得
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # キー名の調整（必要に応じて）
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        フレームを前処理してモデル入力形式に変換
        
        Args:
            frame: 入力フレーム (BGR format)
            
        Returns:
            前処理済みテンソル
        """
        # BGRからRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # リサイズ
        frame_resized = cv2.resize(frame_rgb, self.input_size)
        
        # 正規化 [0, 255] -> [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # チャンネル次元を最初に移動 (H, W, C) -> (C, H, W)
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        
        # テンソルに変換してバッチ次元を追加
        tensor = torch.from_numpy(frame_transposed).unsqueeze(0).to(self.device)
        
        return tensor

    def _predict_frame(self, frame: np.ndarray) -> float:
        """
        単一フレームの「本物らしさ」スコアを予測
        
        Args:
            frame: 入力フレーム
            
        Returns:
            本物らしさスコア (0.0 - 1.0)
        """
        with torch.no_grad():
            # 前処理
            input_tensor = self._preprocess_frame(frame)
            
            # 推論
            outputs = self.model(input_tensor)
            
            # ソフトマックスで確率に変換
            probs = torch.softmax(outputs, dim=1)
            
            # 「本物」クラスの確率を取得 (クラス1が本物と仮定)
            real_score = probs[0, 1].item()
            
            return real_score

    def check(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        フレームリストを解析し、強化されたパッシブ検知を実行
        
        Args:
            frames: フレームのリスト
            
        Returns:
            検知結果の辞書
        """
        if not frames:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "No frames to analyze",
                "enhanced_analysis": {
                    "texture_analysis": None,
                    "identity_analysis": None,
                    "virtual_camera_analysis": None
                }
            }
        
        # 1. 既存のSilent-Face-Anti-Spoofingチェック
        sample_frames = self._sample_frames(frames)
        scores = []
        
        for frame in sample_frames:
            try:
                score = self._predict_frame(frame)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error processing frame with MiniFASNetV2: {e}")
                continue
        
        if not scores:
            base_score = 0.0
            base_passed = False
        else:
            base_score = float(np.mean(scores))
            base_passed = base_score >= self.threshold
        
        # 2. 新しい検知器による強化分析
        logger.info("Starting enhanced passive analysis...")
        
        # 検知器1: テクスチャ・ノイズ分析
        logger.info("Running texture and noise analysis...")
        texture_result = self.texture_detector.compute_texture_score(frames)
        
        # 検知器2: 時間軸上の同一性分析
        logger.info("Running face identity consistency analysis...")
        identity_result = self.identity_detector.analyze_face_consistency(frames)
        
        # 検知器3: 入力ソースの信頼性分析
        logger.info("Running virtual camera detection...")
        virtual_camera_result = self.virtual_camera_detector.detect_virtual_camera()
        
        # 3. 総合判定ロジック
        # 各検知器の重み付けスコア計算
        weights = {
            'base_model': 0.4,      # 既存のMiniFASNetV2
            'texture': 0.25,        # テクスチャ分析
            'identity': 0.25,       # 顔一貫性
            'virtual_camera': 0.1   # 仮想カメラ検出
        }
        
        # 正規化されたスコア計算
        normalized_scores = {
            'base_model': base_score,
            'texture': texture_result['average_score'],
            'identity': identity_result['consistency_score'],
            'virtual_camera': 1.0 - virtual_camera_result['risk_score']  # リスクを反転
        }
        
        # 加重平均スコア
        weighted_score = sum(
            normalized_scores[key] * weights[key] 
            for key in weights.keys()
        )
        
        # 個別検知器の失敗チェック
        critical_failures = []
        if not texture_result['passed']:
            critical_failures.append("texture_analysis")
        if not identity_result['passed']:
            critical_failures.append("identity_consistency")
        if not virtual_camera_result['passed']:
            critical_failures.append("virtual_camera_detected")
        
        # 最終判定
        # 1つでも重要な失敗があれば、加重スコアが高くても失敗とする
        if critical_failures and weighted_score < 0.8:
            final_passed = False
            final_message = f"Enhanced passive check failed: {', '.join(critical_failures)}"
        elif weighted_score >= 0.6:  # 緩和された閾値
            final_passed = True
            final_message = "Enhanced passive check passed"
        else:
            final_passed = False
            final_message = f"Enhanced passive check failed (weighted score: {weighted_score:.3f})"
        
        return {
            "passed": bool(final_passed),
            "average_real_score": round(float(weighted_score), 3),
            "base_model_score": round(float(base_score), 3),
            "frame_scores": [round(float(s), 3) for s in scores],
            "message": str(final_message),
            "enhanced_analysis": {
                "texture_analysis": texture_result,
                "identity_analysis": identity_result,
                "virtual_camera_analysis": virtual_camera_result,
                "weighted_scores": {k: round(float(v), 3) for k, v in normalized_scores.items()},
                "weights_used": weights,
                "critical_failures": critical_failures
            }
        }

    def _sample_frames(self, frames: List[np.ndarray], max_frames: int = 10) -> List[np.ndarray]:
        """
        フレームをサンプリング（計算量削減のため）
        
        Args:
            frames: 全フレーム
            max_frames: 最大サンプリング数
            
        Returns:
            サンプリングされたフレーム
        """
        if len(frames) <= max_frames:
            return frames
        
        # 均等にサンプリング
        step = len(frames) // max_frames
        sampled_frames = []
        for i in range(0, len(frames), step):
            if len(sampled_frames) >= max_frames:
                break
            sampled_frames.append(frames[i])
        
        return sampled_frames
