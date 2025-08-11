# 生体検知API

ディープフェイクや写真・動画によるなりすまし攻撃を防ぐ生体検知APIです。

## 特徴

- **ダブルチェック機構**: アクティブ方式とパッシブ方式を組み合わせた堅牢な検知
- **アクティブ検知**: MediaPipeを使用したチャレンジシーケンス（2回まばたき→左に首をかしげる→右に首をかしげる）の検証
- **強化されたパッシブ検知**: 
  - **検知器1**: Silent-Face-Anti-Spoofingモデル（MiniFASNetV2）によるディープフェイク検出
  - **検知器2**: テクスチャ・ノイズ分析による加工痕跡の検出
  - **検知器3**: 時間軸上の顔同一性分析（DeepFace）によるフレーム間一貫性検証
  - **検知器4**: 入力ソース信頼性分析による仮想カメラ検出
- **Dockerコンテナ**: 高いポータビリティと再現性を確保

## 強化されたパッシブ検知の詳細

### 検知器1: テクスチャ・ノイズ分析
- **LBP (Local Binary Pattern)**: 質感の多様性を分析
- **ハイフリケンシーノイズ**: デジタル加工特有のノイズパターンを検出
- **圧縮アーティファクト**: JPEG圧縮による劣化を分析
- **エッジ一貫性**: エッジの自然さを評価
- **周波数ドメイン分析**: FFTによる周波数異常を検出

### 検知器2: 時間軸上の顔同一性分析
- **DeepFaceライブラリ**: VGG-Faceモデルによる顔特徴抽出
- **フレーム間類似度**: コサイン類似度による一貫性チェック
- **リアルタイム合成検出**: フレーム間の微妙な「顔のブレ」を検知
- **別人混入検出**: 動画内での人物の入れ替わりを検出

### 検知器3: 入力ソース信頼性分析
- **仮想カメラ検出**: v4l2loopback等の仮想デバイスを検出
- **疑わしいプロセス監視**: OBS、ManyCam等の配信・加工ソフトを検出
- **システムレベル分析**: プロセスリストとデバイス情報の総合判定

### 総合判定ロジック
各検知器の結果を重み付けして最終判定：
- MiniFASNetV2: 40%
- テクスチャ分析: 25%
- 顔同一性分析: 25%
- 仮想カメラ検出: 10%

## セットアップ

### 1. モデルファイルの準備

```bash
# モデルファイルをダウンロード（オプション）
python download_model.py
```

または、手動で`2.7_80x80_MiniFASNetV2.pth`を`app/models/`ディレクトリに配置してください。

### 2. Docker環境での起動

```bash
# APIサーバーを起動
docker compose up --build

# バックグラウンドで起動
docker compose up -d --build
```

### 3. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# Swagger UI でAPI仕様を確認
# ブラウザで http://localhost:8000/docs にアクセス
```

## API仕様

### エンドポイント: `POST /liveness/check`

#### リクエスト
- **Content-Type**: `multipart/form-data`
- **パラメータ**: `file` (動画ファイル)
- **制限**:
  - フォーマット: mp4, mov, mpeg
  - 最大サイズ: 10MB
  - 最大長: 5秒

#### レスポンス例

**成功時 (200 OK)**:
```json
{
    "status": "SUCCESS",
    "reason": "Liveness check passed.",
    "details": {
        "active_check": {
            "passed": true,
            "message": "Challenge sequence completed correctly.",
            "details": {
                "blink_count": 2,
                "tilted_left": true,
                "challenge_completed": true
            }
        },
        "passive_check": {
            "passed": true,
            "average_real_score": 0.95,
            "base_model_score": 0.92,
            "message": "Enhanced passive check passed",
            "enhanced_analysis": {
                "texture_analysis": {
                    "passed": true,
                    "average_score": 0.85,
                    "message": "Texture analysis passed (score: 0.850)"
                },
                "identity_analysis": {
                    "passed": true,
                    "consistency_score": 0.95,
                    "message": "Identity consistency passed (avg: 0.950)"
                },
                "virtual_camera_analysis": {
                    "passed": true,
                    "risk_score": 0.0,
                    "message": "Virtual camera detection passed (risk: 0.000)"
                },
                "weighted_scores": {
                    "base_model": 0.92,
                    "texture": 0.85,
                    "identity": 0.95,
                    "virtual_camera": 1.0
                },
                "critical_failures": []
            }
        }
    },
    "video_info": {
        "duration": 4.2,
        "frame_count": 126,
        "fps": 30
    }
}
```

**失敗時 (200 OK)**:
```json
{
    "status": "FAILURE",
    "reason": "Active challenge failed: Insufficient blinks detected: 1/2",
    "details": {
        "active_check": {
            "passed": false,
            "message": "Insufficient blinks detected: 1/2",
            "details": {
                "blink_count": 1,
                "tilted_left": false,
                "challenge_completed": false
            }
        },
        "passive_check": {
            "passed": true,
            "average_real_score": 0.92
        }
    }
}
```

## テスト

### APIテストスクリプト

```bash
# テスト用動画でAPIをテスト
python test_api.py test_video.mp4

# 別のURLでテスト
python test_api.py test_video.mp4 http://your-server:8000
```

### curlでのテスト

```bash
curl -X POST "http://localhost:8000/liveness/check" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_video.mp4"
```

## テストケース

1. **成功ケース**: 開発者自身がチャレンジシーケンス（2回まばたき→左に首をかしげる→右に首をかしげる）を正しく実行した動画
2. **パッシブ失敗（写真攻撃）**: 顔写真をスマートフォンで表示して撮影した動画
3. **パッシブ失敗（ディープフェイク）**: First Order Motion ModelやDeepFaceLiveで生成した動画
4. **パッシブ失敗（仮想カメラ）**: OBSやv4l2loopbackを使用した動画入力
5. **パッシブ失敗（顔入れ替え）**: 動画内で異なる人物の顔が混入した場合
6. **アクティブ失敗**: チャレンジとは異なる動作（順序間違い、まばたき不足、首の傾きが不完全）をした動画

## 開発環境

- **Python**: 3.10
- **フレームワーク**: FastAPI
- **主要ライブラリ**:
  - MediaPipe (顔検出・ランドマーク)
  - PyTorch (Deep Learning)
  - OpenCV (動画処理)
  - DeepFace (顔認識・同一性分析)
  - scikit-image (テクスチャ分析)
  - psutil (システム情報取得)
- **コンテナ**: Docker, Docker Compose

## ファイル構成

```
liveness-api/
├── app/
│   ├── models/              # 学習済みモデル
│   ├── main.py             # FastAPIアプリケーション
│   ├── active_checker.py   # アクティブ検知ロジック
│   ├── passive_checker.py  # パッシブ検知ロジック
│   └── video_processor.py  # 動画処理ユーティリティ
├── Dockerfile              # Dockerイメージ定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
├── download_model.py       # モデルダウンロードスクリプト
├── test_api.py            # APIテストスクリプト
└── README.md              # このファイル
```

## Swagger UI

APIの詳細仕様とインタラクティブなテストは以下で確認できます:
- [http://localhost:8000/docs](http://localhost:8000/docs)

## ログ

アプリケーションのログは標準出力に出力されます:

```bash
# ログを確認
docker compose logs -f liveness-api
```

## トラブルシューティング

### モデルファイルが見つからない場合

モックチェッカーが使用され、常に成功を返します。本格運用前にモデルファイルを配置してください。

### MediaPipeエラー

MediaPipeの初期化に失敗した場合、モックチェッカーが使用されます。

### メモリ不足

大きな動画ファイルでメモリ不足が発生する場合は、フレーム制限を調整してください（`max_frames`パラメータ）。
