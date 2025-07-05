# 檔案: ingestVideos.py

import os
import pandas as pd
import httpx
from typing import Dict, Optional, Tuple

# 從我們新的「資料庫中心」匯入所有需要的東西
try:
    from databaseSetup import SessionLocal, PitchRecording, Kinematics
except ImportError:
    print("❌ 錯誤: 找不到 database_setup.py。請確認檔案存在且架構正確。")
    exit()

# 從 KinematicsModulev2.py 匯入特徵提取函式
try:
    from kinematicsModule import extract_pitching_biomechanics
except ImportError:
    print("❌ 錯誤: 找不到 KinematicsModulev2.py。請確認檔案存在。")
    exit()


def analyze_video_and_get_features(video_path: str, pose_api_url: str) -> Optional[Tuple[Dict, Dict]]:
    """
    分析單一影片檔案，呼叫 Pose API，並回傳 (原始pose_data, 計算後的features) 的元組。
    """
    if not os.path.exists(video_path):
        print(f"  ❌ 錯誤：找不到影片檔案 {video_path}")
        return None, None

    try:
        # 讀取影片檔案
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        # 使用同步的 httpx.Client 進行 API 請求
        with httpx.Client(timeout=300.0) as client:
            files = {"file": (os.path.basename(video_path), video_bytes, "video/mp4")}
            print(f"  ... 正在呼叫 Pose API: {pose_api_url}")
            response = client.post(pose_api_url, files=files)
            response.raise_for_status() 
            
            pose_data = response.json()
            features = extract_pitching_biomechanics(pose_data)
            
            # 驗證特徵，確保所有值都不是 None
            if features and all(value is not None for value in features.values()):
                return pose_data, features 
            else:
                print(f"  ⚠️ 從 API 回應中無法提取有效特徵。特徵值: {features}")
                return None, None

    except httpx.RequestError as e:
        print(f"  ❌ 網路錯誤：無法連接到 Pose API {e.request.url}。請確認 API 服務正在運行。")
    except httpx.HTTPStatusError as e:
        print(f"  ❌ API 錯誤：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"  ❌ 在分析影片 '{os.path.basename(video_path)}' 時發生未知錯誤: {e}")
        
    return None, None


if __name__ == "__main__":
    # --- 1. 參數設定 ---
    # 這裡集中了所有您可以修改的參數
    PITCHER_FOLDER_NAME = "Shohei_Ohtani_FS_videos_4S"
    CSV_FILENAME = "Shohei_Ohtani_FS.csv"
    FILENAME_COLUMN_IN_CSV = "Filename"  # CSV 中代表影片檔名的欄位名稱
    DATA_ROOT_FOLDER = "data"
    POSE_API_URL = "http://localhost:8000/pose_video"
    
    # 設定本次最多處理幾筆「新」影片 (設定為 None 則不限制)
    MAX_VIDEOS_TO_PROCESS = None
    
    # --- 程式會自動組合出完整路徑 ---
    DATA_DIRECTORY = os.path.join("..", DATA_ROOT_FOLDER, PITCHER_FOLDER_NAME)

    # --- 2. 讀取 CSV ---
    csv_path = os.path.join(DATA_DIRECTORY, CSV_FILENAME)
    if not os.path.exists(csv_path):
        print(f"❌ 錯誤：找不到 CSV 檔案 {csv_path}")
        exit()
    
    video_metadata_df = pd.read_csv(csv_path)
    try:
        metadata_map = video_metadata_df.set_index(FILENAME_COLUMN_IN_CSV).to_dict('index')
    except KeyError:
        print(f"❌ 嚴重錯誤：您設定的檔名欄位 '{FILENAME_COLUMN_IN_CSV}' 不存在於 CSV 檔案中。")
        print(f"   偵測到的實際欄位為: {video_metadata_df.columns.tolist()}")
        exit()

    # --- 3. 遍歷、檢查、分析、儲存 ---
    db = SessionLocal()
    try:
        all_videos_in_folder = [f for f in os.listdir(DATA_DIRECTORY) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        total_videos = len(all_videos_in_folder)
        new_records_count = 0

        for i, video_name in enumerate(all_videos_in_folder):
            print(f"\n[{i+1}/{total_videos}] 正在檢查影片: {video_name}")

            # 檢查是否已達處理上限
            if MAX_VIDEOS_TO_PROCESS is not None and new_records_count >= MAX_VIDEOS_TO_PROCESS:
                print(f"🔵 已達到本次設定的處理上限 ({MAX_VIDEOS_TO_PROCESS} 筆)，提前結束。")
                break

            # 檢查重複
            existing_record = db.query(PitchRecording).filter(PitchRecording.video_filename == video_name).first()
            if existing_record:
                print(f"  ⏭️  紀錄已存在，跳過。")
                continue

            # 分析影片
            pose_data, features = analyze_video_and_get_features(os.path.join(DATA_DIRECTORY, video_name), POSE_API_URL)
            
            if pose_data and features:
                video_info = metadata_map.get(video_name, {})

                # 步驟 A: 儲存「原始紀錄」到 pitch_record 表
                new_pitch_record = PitchRecording(
                    player_name=video_info.get('player_name'),
                    pitch_type=video_info.get('pitch_type'),
                    video_filename=video_name,
                    description=video_info.get('description'),
                    source_csv=CSV_FILENAME,
                    keypoints_data=pose_data
                )
                db.add(new_pitch_record)
                db.commit()
                db.refresh(new_pitch_record)

                # 步驟 B: 儲存「運動學特徵」到 kinematics 表，並關聯
                new_kinematics_obj = Kinematics(
                    pitch_record_id=new_pitch_record.id,
                    trunk_flexion_excursion=features.get('Trunk_flexion_excursion'),
                    pelvis_obliquity_at_fc=features.get('Pelvis_obliquity_at_FC'),
                    trunk_rotation_at_br=features.get('Trunk_rotation_at_BR'),
                    shoulder_abduction_at_br=features.get('Shoulder_abduction_at_BR'),
                    trunk_flexion_at_br=features.get('Trunk_flexion_at_BR'),
                    trunk_lateral_flexion_at_hs=features.get('Trunk_lateral_flexion_at_HS'),
                    release_frame=features.get('release_frame'),
                    landing_frame=features.get('landing_frame'),
                    shoulder_frame=features.get('shoulder_frame'),
                    total_frames=features.get('total_frames')
                )
                db.add(new_kinematics_obj)
                db.commit()
                
                new_records_count += 1
                print(f"  ✅ 新紀錄與特徵已成功存入資料庫。")
            else:
                print(f"  ❌ 分析失敗或未提取到有效特徵，已跳過。")

    except Exception as e:
        print(f"\n處理過程中發生嚴重錯誤: {e}")
        db.rollback()
    finally:
        db.close()

    print("\n========================================================")
    print(f"批次分析執行完畢。本次新增了 {new_records_count} 筆新資料。")