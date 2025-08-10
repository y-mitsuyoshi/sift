# 高精度eKYC生体検知API

## セットアップ

1. モデルファイル（2.7_80x80_MiniFASNetV2.pth）を`app/models`に配置してください。
2. `docker compose up`でAPIサーバーを起動します。

## API仕様

- エンドポイント: `POST /liveness/check`
- リクエスト: `multipart/form-data`で動画ファイル（mp4, 5秒以内, 最大10MB）を`file`として送信
- レスポンス: 判定結果JSON（詳細は要件定義参照）

## 開発環境
- Python 3.10
- FastAPI
- Docker, Docker Compose

## Swagger UI
- [http://localhost:8000/docs](http://localhost:8000/docs)
