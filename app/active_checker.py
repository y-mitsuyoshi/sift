import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import math

class ActiveChecker:
    def __init__(self):
        """
        アクティブ検知器の初期化
        MediaPipeの顔検出とランドマーク検出を初期化
        """
        # MediaPipe初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 状態管理
        self.reset_state()
        
        # まばたき検出用のランドマークインデックス
        # 左目: [362, 385, 387, 263, 373, 380]
        # 右目: [33, 160, 158, 133, 153, 144]
        self.left_eye_indices = [362, 385, 387, 263, 373, 380]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # 顔の向き検出用のランドマークインデックス
        # 鼻先、左右の顔輪郭点
        self.nose_tip_index = 1
        self.left_face_index = 234
        self.right_face_index = 454
        
        # しきい値設定
        self.blink_threshold = 0.25  # まばたき判定のEAR閾値
        self.head_turn_threshold = 0.15  # 顔の向き判定の閾値

    def reset_state(self):
        """
        状態をリセット
        """
        self.blink_count = 0
        self.is_eye_closed = False
        self.has_turned_left = False
        self.has_turned_right = False
        self.challenge_completed = False
        self.frame_count = 0
        self.head_positions = []

    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Eye Aspect Ratio (EAR) を計算
        まばたき検出に使用
        
        Args:
            eye_landmarks: 目のランドマーク座標
            
        Returns:
            EAR値
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # 垂直距離
        vertical_1 = math.sqrt((eye_landmarks[1][0] - eye_landmarks[5][0])**2 + 
                              (eye_landmarks[1][1] - eye_landmarks[5][1])**2)
        vertical_2 = math.sqrt((eye_landmarks[2][0] - eye_landmarks[4][0])**2 + 
                              (eye_landmarks[2][1] - eye_landmarks[4][1])**2)
        
        # 水平距離
        horizontal = math.sqrt((eye_landmarks[0][0] - eye_landmarks[3][0])**2 + 
                              (eye_landmarks[0][1] - eye_landmarks[3][1])**2)
        
        # EAR計算
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _detect_blink(self, landmarks) -> bool:
        """
        まばたきを検出
        
        Args:
            landmarks: 顔のランドマーク
            
        Returns:
            まばたきが検出されたかどうか
        """
        # 左目のランドマーク取得
        left_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                   for i in self.left_eye_indices]
        
        # 右目のランドマーク取得
        right_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                    for i in self.right_eye_indices]
        
        # 両目のEARを計算
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # 平均EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # まばたき検出ロジック
        blink_detected = False
        
        if avg_ear < self.blink_threshold:
            if not self.is_eye_closed:
                self.is_eye_closed = True
        else:
            if self.is_eye_closed:
                # 目が開いた瞬間 = まばたき完了
                self.blink_count += 1
                self.is_eye_closed = False
                blink_detected = True
        
        return blink_detected

    def _detect_head_turn(self, landmarks) -> str:
        """
        顔の向きを検出
        
        Args:
            landmarks: 顔のランドマーク
            
        Returns:
            顔の向き ('left', 'right', 'center')
        """
        # 鼻先と左右の顔輪郭点を取得
        nose_tip = landmarks.landmark[self.nose_tip_index]
        left_face = landmarks.landmark[self.left_face_index]
        right_face = landmarks.landmark[self.right_face_index]
        
        # 顔の中心からの鼻先の相対位置を計算
        face_center_x = (left_face.x + right_face.x) / 2
        nose_relative_x = nose_tip.x - face_center_x
        
        # 顔の幅で正規化
        face_width = abs(right_face.x - left_face.x)
        if face_width == 0:
            return 'center'
        
        normalized_position = nose_relative_x / face_width
        
        # 顔の向き判定
        if normalized_position < -self.head_turn_threshold:
            return 'left'
        elif normalized_position > self.head_turn_threshold:
            return 'right'
        else:
            return 'center'

    def _process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        単一フレームを処理
        
        Args:
            frame: 入力フレーム
            
        Returns:
            処理結果
        """
        self.frame_count += 1
        
        # RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 顔のランドマーク検出
        results = self.face_mesh.process(rgb_frame)
        
        frame_result = {
            'frame_number': self.frame_count,
            'face_detected': False,
            'blink_detected': False,
            'head_direction': 'none',
            'blink_count': self.blink_count,
            'challenge_status': 'in_progress'
        }
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame_result['face_detected'] = True
                
                # まばたき検出
                blink_detected = self._detect_blink(face_landmarks)
                frame_result['blink_detected'] = blink_detected
                
                # 顔の向き検出
                head_direction = self._detect_head_turn(face_landmarks)
                frame_result['head_direction'] = head_direction
                
                # チャレンジシーケンスの進行チェック
                if self.blink_count >= 2 and not self.has_turned_left:
                    # 2回まばたき完了後、左向きを待つ
                    if head_direction == 'left':
                        self.has_turned_left = True
                        frame_result['challenge_status'] = 'left_completed'
                elif self.blink_count >= 2 and self.has_turned_left and not self.has_turned_right:
                    # 左向き完了後、右向きを待つ
                    if head_direction == 'right':
                        self.has_turned_right = True
                        self.challenge_completed = True
                        frame_result['challenge_status'] = 'completed'
                
                break  # 最初の顔のみ処理
        
        frame_result['blink_count'] = self.blink_count
        return frame_result

    def check(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        チャレンジシーケンス（2回まばたき→左を向く→右を向く）判定
        
        Args:
            frames: フレームのリスト
            
        Returns:
            判定結果
        """
        # 状態リセット
        self.reset_state()
        
        if not frames:
            return {
                "passed": False,
                "message": "No frames to analyze",
                "details": {
                    "blink_count": 0,
                    "turned_left": False,
                    "turned_right": False,
                    "challenge_completed": False
                }
            }
        
        frame_results = []
        
        # 全フレームを処理
        for frame in frames:
            try:
                result = self._process_frame(frame)
                frame_results.append(result)
                
                # チャレンジ完了の早期終了
                if self.challenge_completed:
                    break
                    
            except Exception as e:
                print(f"Error processing frame {self.frame_count}: {e}")
                continue
        
        # 最終判定
        success_conditions = {
            "sufficient_blinks": self.blink_count >= 2,
            "turned_left": self.has_turned_left,
            "turned_right": self.has_turned_right,
            "challenge_completed": self.challenge_completed
        }
        
        passed = all(success_conditions.values())
        
        # メッセージ生成
        if passed:
            message = "Challenge sequence completed correctly."
        elif self.blink_count < 2:
            message = f"Insufficient blinks detected: {self.blink_count}/2"
        elif not self.has_turned_left:
            message = "User did not turn left after blinking"
        elif not self.has_turned_right:
            message = "User did not turn right after turning left"
        else:
            message = "Challenge sequence not completed correctly"
        
        return {
            "passed": passed,
            "message": message,
            "details": {
                "blink_count": self.blink_count,
                "turned_left": self.has_turned_left,
                "turned_right": self.has_turned_right,
                "challenge_completed": self.challenge_completed,
                "total_frames_processed": len(frame_results),
                "face_detection_rate": sum(1 for r in frame_results if r['face_detected']) / len(frame_results) if frame_results else 0
            },
            "frame_analysis": frame_results[-10:] if len(frame_results) > 10 else frame_results  # 最後の10フレームのみ返す
        }
