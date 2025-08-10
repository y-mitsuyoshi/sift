import numpy as np
from typing import List, Optional

class PassiveChecker:
    def __init__(self, model_path: str, threshold: float = 0.8):
        self.model_path = model_path
        self.threshold = threshold
        # TODO: モデルロード処理

    def check(self, frames: List[np.ndarray]) -> dict:
        """
        フレームリストを解析し、平均スコアで判定
        """
        # TODO: モデル推論処理
        # 仮実装
        average_score = 0.95
        passed = average_score >= self.threshold
        return {
            "passed": passed,
            "average_real_score": average_score
        }
