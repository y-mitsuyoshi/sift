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
        
        # まばたき検出用のランドマークインデックス（MediaPipe Face Mesh 468標準）
        # 左目（人から見て左側）: [362, 385, 387, 263, 373, 380]
        # 右目（人から見て右側）: [33, 160, 158, 133, 153, 144]
        self.left_eye_indices = [362, 385, 387, 263, 373, 380]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # 顔の向き検出用のランドマークインデックス
        # 目の外側のポイント（頭部の傾き検出用）
        self.nose_tip_index = 1
        self.left_face_index = 234
        self.right_face_index = 454
        
        # しきい値設定
        self.blink_threshold = 0.18  # まばたき判定のEAR閾値（より寛容に）
        self.head_turn_threshold = 0.03  # 頭部の傾き判定の閾値（度数換算で約0.3度、より敏感に）

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
        
        # 頭の傾き検出のための新しい状態
        self.left_tilt_frames = 0
        self.right_tilt_frames = 0
        self.min_tilt_frames = 3  # 連続3フレーム以上で傾き確定

    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Eye Aspect Ratio (EAR) を計算（簡略版）
        まばたき検出に使用
        
        Args:
            eye_landmarks: 目のランドマーク座標（6点）
            
        Returns:
            EAR値
        """
        if len(eye_landmarks) != 6:
            return 0.3  # デフォルト値（目が開いている状態）
        
        # 垂直距離を2つ計算
        vertical_1 = abs(eye_landmarks[1][1] - eye_landmarks[5][1])
        vertical_2 = abs(eye_landmarks[2][1] - eye_landmarks[4][1])
        
        # 水平距離を計算
        horizontal = abs(eye_landmarks[0][0] - eye_landmarks[3][0])
        
        # EAR計算（ゼロ除算対策）
        if horizontal < 0.001:
            return 0.3
        
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
        
        # デバッグ出力（フレーム毎に10回に1回）
        if self.frame_count % 10 == 0:
            print(f"Frame {self.frame_count}: EAR={avg_ear:.3f}, Threshold={self.blink_threshold}, Closed={self.is_eye_closed}")
        
        # まばたき検出ロジック（改善版）
        blink_detected = False
        
        # EARの変化を追跡してまばたきを検出
        if avg_ear < self.blink_threshold:
            if not self.is_eye_closed:
                self.is_eye_closed = True
        else:
            if self.is_eye_closed:
                # 目が開いた瞬間 = まばたき完了
                self.blink_count += 1
                self.is_eye_closed = False
                blink_detected = True
                print(f"Blink detected! Count: {self.blink_count}, EAR: {avg_ear:.3f}")  # デバッグ用
        
        return blink_detected

    def _detect_head_turn(self, landmarks) -> str:
        """
        頭部の傾きを検出（首をかしげる動作）
        
        Args:
            landmarks: 顔のランドマーク
            
        Returns:
            頭部の傾き ('left', 'right', 'center')
        """
        # 左右の目尻を取得して頭部の傾きを計算
        left_eye_outer = landmarks.landmark[33]   # 左目外側
        right_eye_outer = landmarks.landmark[362] # 右目外側
        
        # 両目を結ぶ線の傾きを計算
        dx = right_eye_outer.x - left_eye_outer.x
        dy = right_eye_outer.y - left_eye_outer.y
        
        if abs(dx) < 0.001:  # ゼロ除算対策
            tilt_angle = 0
        else:
            eye_line_slope = dy / dx
            # 傾き角度を計算（ラジアンから度に変換）
            tilt_angle = math.atan(eye_line_slope) * 180 / math.pi
        
        # デバッグ出力（フレーム毎に5回に1回）
        if self.frame_count % 5 == 0:
            print(f"Head tilt - Frame {self.frame_count}: tilt_angle={tilt_angle:.2f}°, threshold=±{self.head_turn_threshold * 57.3:.1f}°, dx={dx:.3f}, dy={dy:.3f}")
        
        # 頭部の傾き判定（閾値を度数で比較）
        threshold_degrees = self.head_turn_threshold * 57.3  # ラジアンから度への変換係数
        
        if tilt_angle > threshold_degrees:
            return 'left'   # 左にかしげる（時計回り）
        elif tilt_angle < -threshold_degrees:
            return 'right'  # 右にかしげる（反時計回り）
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
                
                # 頭の傾きカウント更新
                if head_direction == 'left':
                    self.left_tilt_frames += 1
                    self.right_tilt_frames = 0  # リセット
                elif head_direction == 'right':
                    self.right_tilt_frames += 1
                    self.left_tilt_frames = 0  # リセット
                else:
                    # centerの場合は両方を減衰させる
                    self.left_tilt_frames = max(0, self.left_tilt_frames - 1)
                    self.right_tilt_frames = max(0, self.right_tilt_frames - 1)
                
                # 連続フレーム数に基づく傾き確定
                left_confirmed = self.left_tilt_frames >= self.min_tilt_frames
                right_confirmed = self.right_tilt_frames >= self.min_tilt_frames
                
                # チャレンジシーケンスの進行チェック
                if self.blink_count >= 2 and not self.has_turned_left:
                    # 2回まばたき完了後、左にかしげるのを待つ
                    if left_confirmed:
                        self.has_turned_left = True
                        frame_result['challenge_status'] = 'left_completed'
                        print(f"Left tilt confirmed at frame {self.frame_count} (consecutive frames: {self.left_tilt_frames})")
                elif self.blink_count >= 2 and self.has_turned_left and not self.has_turned_right:
                    # 左にかしげる完了後、右にかしげるのを待つ
                    if right_confirmed:
                        self.has_turned_right = True
                        self.challenge_completed = True
                        frame_result['challenge_status'] = 'completed'
                        print(f"Right tilt confirmed at frame {self.frame_count} (consecutive frames: {self.right_tilt_frames})")
                
                # デバッグ情報を出力
                if self.frame_count % 10 == 0:
                    print(f"Frame {self.frame_count}: blinks={self.blink_count}, left_frames={self.left_tilt_frames}, right_frames={self.right_tilt_frames}, status={'left_done' if self.has_turned_left else 'waiting_left' if self.blink_count >= 2 else 'waiting_blinks'}")
                
                break  # 最初の顔のみ処理
        
        frame_result['blink_count'] = self.blink_count
        return frame_result

    def check(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        チャレンジシーケンス（2回まばたき→首を左にかしげる→首を右にかしげる）判定
        
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
                    "tilted_left": False,
                    "tilted_right": False,
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
            "tilted_left": self.has_turned_left,
            "tilted_right": self.has_turned_right,
            "challenge_completed": self.challenge_completed
        }
        
        passed = all(success_conditions.values())
        
        # メッセージ生成
        if passed:
            message = "Challenge sequence completed correctly."
        elif self.blink_count < 2:
            message = f"Insufficient blinks detected: {self.blink_count}/2"
        elif not self.has_turned_left:
            message = "User did not tilt head left after blinking"
        elif not self.has_turned_right:
            message = "User did not tilt head right after tilting left"
        else:
            message = "Challenge sequence not completed correctly"
        
        return {
            "passed": passed,
            "message": message,
            "details": {
                "blink_count": self.blink_count,
                "tilted_left": self.has_turned_left,
                "tilted_right": self.has_turned_right,
                "challenge_completed": self.challenge_completed,
                "total_frames_processed": len(frame_results),
                "face_detection_rate": sum(1 for r in frame_results if r['face_detected']) / len(frame_results) if frame_results else 0
            },
            "frame_analysis": frame_results[-10:] if len(frame_results) > 10 else frame_results  # 最後の10フレームのみ返す
        }
