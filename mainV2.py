# 檔案: mainV2.py
# 職責: 作為 API 的入口點，接收請求並完全轉交給服務層處理。

import logging
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
import uvicorn

# --- 從我們的「資料庫中心」和「服務中心」匯入 ---
# 已更新為您最新的駝峰式檔名
import crud
import services
from databaseSetup import get_db, PitchAnalyses
from models import PitchAnalysisUpdate

# --- 全域設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 路由 ---

@app.post("/analyze-pitch/")
async def analyze_pitch(
    db: Session = Depends(get_db),
    video_file: UploadFile = File(...), 
    player_name: str = Form(...),
    benchmark_player_name: Optional[str] = Form(None) 
):
    """
    接收前端請求，將所有工作轉交給服務層，並直接回傳服務層的結果。
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="未上傳影片檔案")

    try:
        
        analysis_result = await services.analyze_pitch_service(
            db=db,
            video_file=video_file,
            player_name=player_name,
            benchmark_player_name=benchmark_player_name
        )
        
        return JSONResponse(content=analysis_result)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"影片分析處理失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"影片分析處理失敗: {str(e)}")


@app.get("/history/")
async def get_history_analyses(pitcher_name: str = None, db: Session = Depends(get_db)):
    # (此路由保留您同事的設計，不變)
    try:
        history_records = crud.get_pitch_analyses(db, pitcher_name)
        return [
            {
                "id": record.id,
                "video_path": record.video_path,
                "max_speed_kmh": record.max_speed_kmh,
                "pitch_score": record.pitch_score,
                "ball_score": record.ball_score,
                "biomechanics_features": record.biomechanics_features,
                "pitcher_name": record.pitcher_name,
                "release_frame_url": record.release_frame_url or "",
                "landing_frame_url": record.landing_frame_url or "",
                "shoulder_frame_url": record.shoulder_frame_url or ""
            }
            for record in history_records
        ]
    except SQLAlchemyError as e:
        logger.error(f"無法獲取歷史紀錄: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法獲取歷史紀錄: {e}")


# @app.delete("/analyses/{analysis_id}")
# async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
#     # (此路由保留您同事的設計，不變)
#     try:
#         if not crud.delete_pitch_analysis(db, analysis_id):
#             raise HTTPException(status_code=404, detail="分析紀錄未找到")
#         logger.info(f"分析紀錄 ID: {analysis_id} 已成功刪除")
#         return {"message": "分析紀錄已成功刪除"}
#     except SQLAlchemyError as e:
#         logger.error(f"刪除分析紀錄失敗: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"刪除分析紀錄失敗: {e}")

# @app.put("/analyses/{analysis_id}")
# async def update_analysis(analysis_id: int, updated_data: PitchAnalysisUpdate, db: Session = Depends(get_db)):
#     # (此路由保留您同事的設計，不變)
#     try:
#         analysis = crud.update_pitch_analysis(db, analysis_id, updated_data)
#         if not analysis:
#             raise HTTPException(status_code=404, detail="分析紀錄未找到")
#         logger.info(f"分析紀錄 ID: {analysis_id} 已成功更新")
#         return analysis
#     except SQLAlchemyError as e:
#         logger.error(f"更新分析紀錄失敗: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"更新分析紀錄失敗: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082)) # 建議使用一個新的埠號
    uvicorn.run("mainV2:app", host="0.0.0.0", port=port, reload=True)
