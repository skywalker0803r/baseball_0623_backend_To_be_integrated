# 檔案: crud.py
# 職責: 作為資料庫的唯一接口 (數據庫管家)，提供所有資料的增刪改查功能。

from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from databaseSetup import PitchAnalyses, PitchModel
from models import PitchAnalysisUpdate


# --- 針對 PitchAnalyses (單次測試結果) 的操作 ---

def get_pitch_analysis(db: Session, analysis_id: int) -> Optional[PitchAnalyses]:
    """根據 ID 獲取單筆分析紀錄。"""
    return db.query(PitchAnalyses).filter(PitchAnalyses.id == analysis_id).first()

def get_pitch_analyses(db: Session, pitcher_name: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[PitchAnalyses]:
    """
    獲取分析紀錄列表，可選擇性地根據投手名稱篩選。
    """
    query = db.query(PitchAnalyses).order_by(PitchAnalyses.id.desc())
    if pitcher_name:
        query = query.filter(PitchAnalyses.pitcher_name == pitcher_name)
    return query.offset(skip).limit(limit).all()

def create_pitch_analysis(db: Session, analysis_data: Dict[str, Any]) -> PitchAnalyses:
    """
    根據傳入的字典，建立一筆新的分析紀錄。
    這個版本更靈活，可以直接接收來自 services 層的字典。
    """
    # 為了對應資料庫模型，我們需要從 scores 字典中提取出 pitch_score 和 ball_score
    scores = analysis_data.get("scores", {})
    db_analysis = PitchAnalyses(
        video_path=analysis_data.get("video_path"),
        pitcher_name=analysis_data.get("pitcher_name"),
        max_speed_kmh=analysis_data.get("max_speed_kmh"),
        pitch_score=scores.get("pitch_mechanics_score"),
        ball_score=scores.get("ball_flight_score"),
        biomechanics_features=analysis_data.get("biomechanics_features"),
        release_frame_url=analysis_data.get("release_frame_url"),
        landing_frame_url=analysis_data.get("landing_frame_url"),
        shoulder_frame_url=analysis_data.get("shoulder_frame_url")
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def update_pitch_analysis(db: Session, analysis_id: int, updated_data: PitchAnalysisUpdate) -> Optional[PitchAnalyses]:
    """更新指定的分析紀錄。"""
    db_analysis = get_pitch_analysis(db, analysis_id)
    if db_analysis:
        # exclude_unset=True 表示只更新前端有提供的欄位
        for key, value in updated_data.dict(exclude_unset=True).items():
            setattr(db_analysis, key, value)
        db.commit()
        db.refresh(db_analysis)
    return db_analysis

def delete_pitch_analysis(db: Session, analysis_id: int) -> bool:
    """刪除指定的分析紀錄。"""
    db_analysis = get_pitch_analysis(db, analysis_id)
    if db_analysis:
        db.delete(db_analysis)
        db.commit()
        return True
    return False


# --- 針對 PitchModel (統計模型) 的操作 ---

def get_pitch_model_by_name(db: Session, model_name: str) -> Optional[PitchModel]:
    """
    根據模型名稱，從 pitch_model 資料表中查詢一個統計模型。
    """
    return db.query(PitchModel).filter(PitchModel.model_name == model_name).first()

