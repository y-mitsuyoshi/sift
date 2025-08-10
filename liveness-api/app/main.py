from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/liveness/check")
async def liveness_check(file: UploadFile = File(...)):
    # TODO: 動画ファイルの検証・処理
    # TODO: パッシブ/アクティブチェック呼び出し
    # TODO: 判定結果をJSONで返却
    return JSONResponse(content={"status": "SUCCESS", "reason": "Stub response.", "details": {}})
