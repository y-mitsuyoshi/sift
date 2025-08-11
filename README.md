# 生体検知API

ディープフェイクや写真・動画によるなりすまし攻撃を防ぐ生体検知APIです。

## 特徴

- **ダブルチェック機構**: アクティブ方式とパッシブ方式を組み合わせた堅牢な検知
- **アクティブ検知**: MediaPipeを使用したチャレンジシーケンス（2回まばたき→左に首をかしげる→右に首をかしげる）の検証
- **パッシブ検知**: Silent-Face-Anti-Spoofingモデルによるなりすまし検出
- **Dockerコンテナ**: 高いポータビリティと再現性を確保

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
            "message": "Passive check passed"
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
2. **パッシブ失敗**: 顔写真をスマートフォンで表示して撮影した動画
3. **アクティブ失敗**: チャレンジとは異なる動作（順序間違い、まばたき不足、首の傾きが不完全）をした動画

## 開発環境

- **Python**: 3.10
- **フレームワーク**: FastAPI
- **主要ライブラリ**:
  - MediaPipe (顔検出・ランドマーク)
  - PyTorch (Deep Learning)
  - OpenCV (動画処理)
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
