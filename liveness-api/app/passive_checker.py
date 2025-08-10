import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import List, Dict, Tuple
import os

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
            
            # 「本物」クラスの確率を取得 (クラス0が本物と仮定)
            real_score = probs[0, 0].item()
            
            return real_score

    def check(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        フレームリストを解析し、パッシブ検知を実行
        
        Args:
            frames: フレームのリスト
            
        Returns:
            検知結果の辞書
        """
        if not frames:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "No frames to analyze"
            }
        
        # 全フレームまたはサンプリングされたフレームで推論
        sample_frames = self._sample_frames(frames)
        scores = []
        
        for frame in sample_frames:
            try:
                score = self._predict_frame(frame)
                scores.append(score)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        if not scores:
            return {
                "passed": False,
                "average_real_score": 0.0,
                "message": "Failed to process any frames"
            }
        
        # 平均スコアを計算
        average_score = float(np.mean(scores))
        passed = average_score >= self.threshold
        
        return {
            "passed": passed,
            "average_real_score": round(average_score, 3),
            "frame_scores": [round(s, 3) for s in scores],
            "message": "Passive check passed" if passed else f"Potential spoof detected (score: {average_score:.3f})"
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
