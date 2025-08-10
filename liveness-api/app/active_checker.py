import numpy as np
from typing import List, Optional

class ActiveChecker:
    def __init__(self):
        # TODO: MediaPipe初期化
        pass

    def check(self, frames: List[np.ndarray]) -> dict:
        """
        チャレンジシーケンス（2回まばたき→左を向く）判定
        """
        # TODO: まばたき・顔向き判定ロジック
        # 仮実装
        passed = True
        message = "Challenge sequence completed correctly."
        return {
            "passed": passed,
            "message": message
        }
