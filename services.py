# 檔案: services.py
# 職責: 處理所有核心商業邏輯，此版本已升級為數據驅動的評分模型。

import os
import shutil
import tempfile
import asyncio
import httpx
import logging
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session
import joblib
import numpy as np

# --- 從我們的「資料庫中心」和「數據庫管家」匯入 ---
import crud
from databaseSetup import PitchModel

# --- 導入其他自定義模組 ---
from kinematicsModule import extract_pitching_biomechanics
from ballClassification import classify_ball_quality
from drawingFunction import render_video_with_pose_and_max_ball_speed

# --- 全域設定 ---
logger = logging.getLogger(__name__)
OUTPUT_VIDEO_DIR = "output_videos"
POSE_API_URL = "http://localhost:8000/pose_video"
BALL_API_URL = "http://localhost:8080/predict"
API_TIMEOUT = 1800.0 

try:
    BALL_PREDICTION_MODEL = joblib.load('random_forest_model.pkl')
    logger.info("✅ 成功載入球路預測模型。")
except Exception as e:
    logger.warning(f"⚠️ 警告：載入 .pkl 模型失敗: {e}。")
    BALL_PREDICTION_MODEL = None

# --- 核心服務函式 ---

async def analyze_video_kinematics(video_bytes: bytes, filename: str) -> Tuple[Dict, Dict]:
    """
    呼叫 Pose API，計算生物力學特徵，並同時回傳原始 pose_data 以供畫圖使用。
    """
    logger.info("服務層：(子任務) 正在呼叫 POSE API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(POSE_API_URL, files=files)
        response.raise_for_status()
        pose_data = response.json()
    logger.info("服務層：(子任務) 正在計算生物力學特徵...")
    biomechanics_features = extract_pitching_biomechanics(pose_data)
    return biomechanics_features, pose_data

async def analyze_ball_flight(video_bytes: bytes, filename: str) -> Dict:
    """
    呼叫 Ball API 以獲取球路相關數據。
    """
    logger.info("服務層：(子任務) 正在呼叫 BALL API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(BALL_API_URL, files=files)
        response.raise_for_status()
        return response.json()

def get_comparison_model(db: Session, benchmark_player_name: str, detected_pitch_type: str) -> Optional[PitchModel]:
    """
    【修改後的輔助函式】
    透過 crud.py 智慧地從資料庫中尋找最適合的比對模型。
    """
    profile_model = None
    
    # 1. 如果偵測到的球種有效，優先嘗試尋找最精準的模型 (投手 + 球種)
    if detected_pitch_type and detected_pitch_type != "Unknown":
        # 組合出與 buildModel.py 完全一致的模型名稱
        ideal_model_name = f"{benchmark_player_name}_{detected_pitch_type}_v1"
        logger.info(f"服務層：正在嘗試載入球種專屬模型: {ideal_model_name}")
        profile_model = crud.get_pitch_model_by_name(db, model_name=ideal_model_name)
        if profile_model:
            return profile_model

    # 2. 如果找不到專屬模型，或球種未知，則嘗試尋找該投手的「通用」模型作為備案
    fallback_model_name = f"{benchmark_player_name}_all_v1"
    logger.warning(f"找不到或未指定專屬模型，嘗試載入通用模型: {fallback_model_name}")
    profile_model = crud.get_pitch_model_by_name(db, model_name=fallback_model_name)

    return profile_model

def calculate_score_from_comparison(features: dict, profile_data: dict) -> int:
    """
    【新的評分函式】
    根據使用者的特徵與標準模型的差距，計算出一個 0-100 的分數。
    差距越小，分數越高。
    """
    if not profile_data:
        return 0 # 如果沒有模型可以比對，分數為 0

    total_score = 0
    feature_count = 0

    for key, user_value in features.items():
        # 只比對在模型中有定義的特徵
        profile_stats = profile_data.get(key.lower())
        if not profile_stats or user_value is None:
            continue
        
        mean = profile_stats.get('mean')
        std = profile_stats.get('std')

        # 確保模型中有 mean 和 std，且 std 不為 0
        if mean is None or std is None or std == 0:
            continue

        # 計算 Z-score，代表偏離了幾個標準差
        z_score = abs((user_value - mean) / std)
        
        # 將 Z-score 轉換為 0-100 的分數
        # 這裡使用一個簡單的轉換：Z-score 為 0 (完全符合平均) 得 100 分
        # Z-score 每增加 1 (偏離一個標準差)，就扣 25 分 (可調整)
        # 最低為 0 分
        feature_score = max(0, 100 - z_score * 25)
        
        total_score += feature_score
        feature_count += 1
    
    if feature_count == 0:
        return 0

    # 回傳所有特徵的平均分數
    final_score = int(total_score / feature_count)
    return final_score


async def analyze_pitch_service(db: Session, video_file, player_name: str, benchmark_player_name: Optional[str] = None) -> Dict:
    """
    【主服務函式】
    執行完整的投球分析、比對、渲染，並將所有結果打包回傳。
    """
    # ... (步驟 1 的臨時檔邏輯不變)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp_video_file:
        temp_video_path = temp_video_file.name
        shutil.copyfileobj(video_file.file, temp_video_file)
    try:
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
        
        # 步驟 2: 並行呼叫 API
        (kinematics_results, ball_data) = await asyncio.gather(
            analyze_video_kinematics(video_bytes, video_file.filename),
            analyze_ball_flight(video_bytes, video_file.filename)
        )
        biomechanics_features, pose_data = kinematics_results
        detected_pitch_type = ball_data.get("predicted_pitch_type", "Unknown")
        
        # 步驟 3: 決定比對標準。如果前端沒傳，就預設為與自己比對。
        comparison_target_name = benchmark_player_name if benchmark_player_name else player_name
        profile_model = get_comparison_model(db, comparison_target_name, detected_pitch_type)
        
        # 步驟 4: 使用新的數據驅動評分邏輯
        pitch_score = 0
        profile_data_for_frontend = None
        if profile_model:
            profile_data_for_frontend = profile_model.profile_data
            pitch_score = calculate_score_from_comparison(
                features=biomechanics_features,
                profile_data=profile_data_for_frontend
            )
        else:
            logger.warning(f"在資料庫中找不到任何可用的比對模型 (比對對象: {comparison_target_name})，pitch_score 將設為 0。")

        ball_score = 0
        if BALL_PREDICTION_MODEL:
            ball_score = classify_ball_quality(ball_data, BALL_PREDICTION_MODEL)
        else:
            logger.warning("球路評分模型未載入，ball_score 將設為 0。")

        # 步驟 5: 渲染影片
        output_filename = f"rendered_{video_file.filename}"
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)
        rendered_video_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
            input_video_path=temp_video_path, pose_json=pose_data,
            ball_json=ball_data, output_video_path=output_video_path
        )
        
        # 步驟 6: 組合最終回傳給 API 層的結果字典
        final_result = {
            "pitcher_name": player_name,
            "output_video_url": rendered_video_path,
            "detected_pitch_type": detected_pitch_type,
            "max_speed_kmh": round(max_speed_kmh, 2),
            "pitch_score": pitch_score,
            "ball_score": ball_score,
            "user_features": biomechanics_features,
            "model_profile": {
                "model_name": profile_model.model_name if profile_model else "N/A",
                "profile_data": profile_data_for_frontend
            },
            "biomechanics_features": biomechanics_features,
            "release_frame_url": f"images/{video_file.filename}_release.jpg",
            "landing_frame_url": f"images/{video_file.filename}_landing.jpg",
            "shoulder_frame_url": f"images/{video_file.filename}_shoulder.jpg"
        }
        return final_result

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
