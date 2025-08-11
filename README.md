# 生体検知API

ディープフェイクや写真・動画によるなりすまし攻撃を防ぐ、動的チャレンジ対応の生体検知APIです。

## 特徴

- **ダブルチェック機構**: アクティブ方式とパッシブ方式を組み合わせた堅牢な検知
- **動的アクティブ検知**: セキュリティを向上させるため、セッションごとにランダムなチャレンジ（顔の向き、まばたき等）を生成します。
  - **チャレンジプール**: `2回まばたき`, `顔を左に向ける`, `顔を右に向ける`, `口を開ける`, `頷く`
- **強化されたパッシブ検知**: 
  - **検知器1**: Silent-Face-Anti-Spoofingモデル（MiniFASNetV2）によるディープフェイク検出
  - **検知器2**: テクスチャ・ノイズ分析による加工痕跡の検出
  - **検知器3**: 時間軸上の顔同一性分析（DeepFace）によるフレーム間一貫性検証
  - **検知器4**: 入力ソース信頼性分析による仮想カメラ検出
- **Dockerコンテナ**: 高いポータビリティと再現性を確保

## API仕様

生体検知はセッションベースの2ステッププロセスで行われます。

### ステップ1: セッションの開始

まず、クライアントは新しい生体検知セッションを開始して、そのセッションで実行すべきチャレンジのリストを取得します。

#### エンドポイント: `POST /liveness/session/start`

**リクエストボディ**: なし

**レスポンス例 (200 OK)**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "challenges": [
    "顔を左に向けてください",
    "口を開けてください"
  ]
}
```
- `session_id`: このセッションを一位に識別するID。ステップ2で必要になります。
- `challenges`: ユーザーに提示すべき、実行するアクションの指示リスト（日本語）。

---

### ステップ2: 動画を送信して確認

ユーザーにチャレンジを実行してもらい、その様子を撮影した動画を `session_id` と共に送信します。

#### エンドポイント: `POST /liveness/check`

**リクエスト (`multipart/form-data`)**:
- `session_id`: (string) ステップ1で取得したセッションID
- `file`: (video file) ユーザーがチャレンジを実行している動画ファイル

**制限**:
- フォーマット: mp4, mov, mpeg
- 最大長: 15秒

**レスポンス例 (成功時)**:
```json
{
    "status": "SUCCESS",
    "reason": "Active check passed, passive check indicates low risk.",
    "details": {
        "active_check": {
            "passed": true,
            "message": "Challenge sequence completed successfully.",
            "details": {
                "challenges_requested": [
                    "TURN_LEFT",
                    "OPEN_MOUTH"
                ],
                "challenges_completed": 2,
                "total_frames_processed": 150
            }
        },
        "passive_check": {
            "passed": true,
            "average_real_score": 1.0,
            "message": "Fallback passive checker used. Check skipped."
        }
    },
    "video_info": {
        "duration": 5.0,
        "frame_count": 150,
        "fps": 30
    }
}
```

**レスポンス例 (失敗時)**:
```json
{
    "status": "FAILURE",
    "reason": "Active challenge failed: Challenge failed at step 1: Did not detect '顔を左に向けてください'.",
    "details": {
        "active_check": {
            "passed": false,
            "message": "Challenge failed at step 1: Did not detect '顔を左に向けてください'.",
            "details": {
                "challenges_requested": [
                    "TURN_LEFT",
                    "OPEN_MOUTH"
                ],
                "challenges_completed": 0,
                "total_frames_processed": 225
            }
        },
        "passive_check": {
            "passed": true,
            "average_real_score": 1.0,
            "message": "Fallback passive checker used. Check skipped."
        }
    }
}
```

## テスト

### curlでのテスト

1. **セッションを開始して `session_id` を取得:**
```bash
SESSION_ID=$(curl -X POST http://localhost:8000/liveness/session/start | jq -r '.session_id')
echo "Session started: $SESSION_ID"
```
*(jqが必要です)*

2. **取得した `session_id` を使って動画を送信:**
```bash
curl -X POST "http://localhost:8000/liveness/check" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "session_id=$SESSION_ID" \
     -F "file=@/path/to/your/video.mp4"
```

### テストケース
1. **成功ケース**: `session/start` で取得したチャレンジを正しく実行した動画。
2. **アクティブ失敗**: チャレンジとは異なる動作（順序間違い、まばたき不足など）をした動画。
3. **パッシブ失敗**: 写真攻撃、ディープフェイク動画など。

---
(以降のセクションは変更なし)

## 強化されたパッシブ検知の詳細
... (省略) ...
