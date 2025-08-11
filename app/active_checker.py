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
        # 両目の外側のポイント（頭部の傾き検出用）
        self.nose_tip_index = 1
        self.left_face_index = 234
        self.right_face_index = 454
        
        # しきい値設定
        self.blink_threshold = 0.12  # まばたき判定のEAR閾値（より厳しく調整）
        self.head_turn_threshold = 0.15  # 頭部の傾き判定の閾値（度数換算で約8.6度、より検出しやすく）

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
        self.min_tilt_frames = 2  # 連続2フレーム以上で傾き確定（より敏感に）
        
        # まばたき検出の改善用
        self.closed_frames = 0
        self.min_closed_frames = 3  # 連続3フレーム以上目が閉じている状態でまばたきと判定（より厳しく）
        
        # まばたき間隔制御（連続まばたきを防ぐ）
        self.last_blink_frame = 0
        self.min_blink_interval = 15  # まばたき間の最小フレーム数（約0.5秒間隔）
        
        # 角度追跡（右の傾き検出のため）
        self.previous_angles = []
        self.max_angle_history = 10  # 過去10フレームの角度を保持

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
        
        # EARの変化を追跡してまばたきを検出（連続フレーム要件追加）
        if avg_ear < self.blink_threshold:
            self.closed_frames += 1
            if not self.is_eye_closed and self.closed_frames >= self.min_closed_frames:
                self.is_eye_closed = True
        else:
            if (self.is_eye_closed and self.closed_frames >= self.min_closed_frames and 
                self.frame_count - self.last_blink_frame >= self.min_blink_interval):
                # 目が開いた瞬間 = まばたき完了（連続で閉じていた場合のみ、間隔制限あり）
                self.blink_count += 1
                blink_detected = True
                self.last_blink_frame = self.frame_count
                print(f"Blink detected! Count: {self.blink_count}, EAR: {avg_ear:.3f}, Closed frames: {self.closed_frames}, Frame: {self.frame_count}")
            
            # リセット
            self.is_eye_closed = False
            self.closed_frames = 0
        
        return blink_detected

    def _detect_head_turn(self, landmarks) -> str:
        """
        頭部の傾きを検出（首をかしげる動作）
        
        Args:
            landmarks: 顔のランドマーク
            
        Returns:
            頭部の傾き ('left', 'right', 'center')
        """
        # より確実な頭部傾き検出のため複数の方法を試す
        
        # 方法1: 両目の外側ポイントを使用（MediaPipe座標系）
        left_eye_outer = landmarks.landmark[33]   # 左目外側
        right_eye_outer = landmarks.landmark[263] # 右目外側
        
        # 両目を結ぶ線の傾きを計算（画像座標系で考える）
        dx = right_eye_outer.x - left_eye_outer.x
        dy = right_eye_outer.y - left_eye_outer.y
        
        # 傾き角度を計算（atan2で-180°～180°の範囲）
        tilt_angle_degrees = math.atan2(dy, dx) * 180 / math.pi
        
        # 方法2: 眉毛と顎のポイントでも確認（より確実な検出のため）
        # 左眉毛の端と右眉毛の端
        left_eyebrow = landmarks.landmark[46]   # 左眉毛
        right_eyebrow = landmarks.landmark[276] # 右眉毛
        
        eyebrow_dx = right_eyebrow.x - left_eyebrow.x
        eyebrow_dy = right_eyebrow.y - left_eyebrow.y
        eyebrow_angle = math.atan2(eyebrow_dy, eyebrow_dx) * 180 / math.pi
        
        # 方法3: 鼻先と顔の中央線を使用
        nose_tip = landmarks.landmark[1]      # 鼻先
        chin = landmarks.landmark[18]         # 顎
        face_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        
        # 顔の中央からの鼻の偏差で傾きを判定
        nose_deviation = nose_tip.x - face_center_x
        
        # デバッグ出力（フレーム毎に3回に1回）
        if self.frame_count % 3 == 0:
            print(f"Head tilt analysis - Frame {self.frame_count}:")
            print(f"  Eyes: tilt={tilt_angle_degrees:.2f}°, dx={dx:.3f}, dy={dy:.3f}")
            print(f"  Eyebrows: tilt={eyebrow_angle:.2f}°")
            print(f"  Nose deviation: {nose_deviation:.3f}")
        
        # 複数の指標を統合して判定
        threshold_degrees = 8.0  # より大きなしきい値に調整
        nose_threshold = 0.02    # 鼻の偏差しきい値
        
        # 判定ロジック（複数の方法を組み合わせ）
        eye_left = tilt_angle_degrees > threshold_degrees
        eye_right = tilt_angle_degrees < -threshold_degrees
        eyebrow_left = eyebrow_angle > threshold_degrees
        eyebrow_right = eyebrow_angle < -threshold_degrees
        nose_left = nose_deviation < -nose_threshold  # 鼻が左に偏った = 左傾き
        nose_right = nose_deviation > nose_threshold  # 鼻が右に偏った = 右傾き
        
        # 投票による判定
        left_votes = sum([eye_left, eyebrow_left, nose_left])
        right_votes = sum([eye_right, eyebrow_right, nose_right])
        
        if left_votes >= 2:
            result = 'left'
        elif right_votes >= 2:
            result = 'right'
        elif left_votes >= 1 or right_votes >= 1:
            # 1票でも決定する場合（より敏感に）
            if left_votes > right_votes:
                result = 'left'
            elif right_votes > left_votes:
                result = 'right'
            else:
                result = 'center'
        else:
            result = 'center'
        
        # デバッグ出力
        if self.frame_count % 3 == 0:
            print(f"  Votes: left={left_votes}, right={right_votes} -> {result}")
        
        return result

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
            'left_turn_detected': False,
            'right_turn_detected': False,
            'left_turn_status': self.has_turned_left,
            'right_turn_status': self.has_turned_right,
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
                    print(f"Left tilt detected - frames: {self.left_tilt_frames}")
                elif head_direction == 'right':
                    self.right_tilt_frames += 1
                    self.left_tilt_frames = 0  # リセット
                    print(f"Right tilt detected - frames: {self.right_tilt_frames}")
                else:
                    # どちらでもない場合はカウンターを減らす
                    if self.left_tilt_frames > 0:
                        self.left_tilt_frames = max(0, self.left_tilt_frames - 1)
                    if self.right_tilt_frames > 0:
                        self.right_tilt_frames = max(0, self.right_tilt_frames - 1)
                
                # 左傾きの検出（まばたき2回完了後かつ左傾きがまだの場合）
                if (self.blink_count >= 2 and not self.has_turned_left and 
                    self.left_tilt_frames >= self.min_tilt_frames):
                    self.has_turned_left = True
                    frame_result['left_turn_detected'] = True
                    frame_result['challenge_status'] = 'left_completed'
                    print(f"LEFT HEAD TILT COMPLETED! (frames: {self.left_tilt_frames})")
                
                # 右傾きの検出（左傾き完了後）
                if (self.has_turned_left and not self.has_turned_right and 
                    self.right_tilt_frames >= self.min_tilt_frames):
                    self.has_turned_right = True
                    frame_result['right_turn_detected'] = True
                    self.challenge_completed = True
                    frame_result['challenge_status'] = 'completed'
                    print(f"RIGHT HEAD TILT COMPLETED! (frames: {self.right_tilt_frames}) - CHALLENGE COMPLETE!")
                
                # フレーム結果のステータス更新
                frame_result['left_turn_status'] = self.has_turned_left
                frame_result['right_turn_status'] = self.has_turned_right
                
                # チャレンジ完了後のステータス維持
                if self.challenge_completed:
                    frame_result['challenge_status'] = 'completed'
                elif self.has_turned_left and self.blink_count >= 2:
                    frame_result['challenge_status'] = 'waiting_right'
                elif self.blink_count >= 2:
                    frame_result['challenge_status'] = 'waiting_left'
                else:
                    frame_result['challenge_status'] = 'waiting_blinks'
                
                # デバッグ情報を出力
                if self.frame_count % 10 == 0:
                    print(f"Frame {self.frame_count}: blinks={self.blink_count}, left_frames={self.left_tilt_frames}, right_frames={self.right_tilt_frames}, status={'right_done' if self.has_turned_right else 'waiting_right' if self.has_turned_left else 'waiting_left' if self.blink_count >= 2 else 'waiting_blinks'}")
                
                break  # 最初の顔のみ処理
        
        frame_result['blink_count'] = self.blink_count
        return frame_result

    def check(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        チャレンジシーケンス（2回まばたき→左に首をかしげる→右に首をかしげる）判定
        
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
                    "left_turn_detected": False,
                    "right_turn_detected": False,
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
                "left_turn_detected": self.has_turned_left,  # APIレスポンス形式に合わせて修正
                "right_turn_detected": self.has_turned_right,  # APIレスポンス形式に合わせて修正
                "challenge_completed": self.challenge_completed,
                "total_frames_processed": len(frame_results),
                "face_detection_rate": sum(1 for r in frame_results if r['face_detected']) / len(frame_results) if frame_results else 0
            },
            "frame_analysis": frame_results[-10:] if len(frame_results) > 10 else frame_results  # 最後の10フレームのみ返す
        }
