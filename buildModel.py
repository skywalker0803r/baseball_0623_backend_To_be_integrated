# 檔案: buildModel.py
# 職責: 從資料庫根據指定條件撈取訓練數據，建立統計模型，並將模型存回資料庫。

import json
import pandas as pd
from sqlalchemy import or_, and_
from databaseSetup import SessionLocal, PitchRecording, Kinematics, PitchModel

def create_pitch_profile(features_data: list) -> dict:
    """
    從一系列特徵資料中，為每個特徵建立一個統計模型 (使用百分位數法)。
    """
    if not features_data:
        print("警告：沒有提供任何特徵資料，無法建立模型。")
        return {}
    
    df = pd.DataFrame(features_data)
    
    profile = {}
    feature_columns = [
        'trunk_flexion_excursion', 'pelvis_obliquity_at_fc',
        'trunk_rotation_at_br', 'shoulder_abduction_at_br',
        'trunk_flexion_at_br', 'trunk_lateral_flexion_at_hs',
        'release_frame', 'landing_frame', 
        'shoulder_frame', 'total_frames'
    ]
    
    for col_name in feature_columns:
        if col_name not in df.columns:
            continue
            
        valid_values = df[col_name].dropna()
        
        if len(valid_values) < 2:
            print(f"警告：特徵 '{col_name}' 的有效數據不足，已跳過。")
            continue

        p10 = valid_values.quantile(0.10)
        p90 = valid_values.quantile(0.90)
        
        stats = {
            'min': round(p10, 4),
            'max': round(p90, 4),
            'p10': round(p10, 4),
            'p50_median': round(valid_values.median(), 4),
            'p90': round(p90, 4),
            'mean': round(valid_values.mean(), 4),
            'std': round(valid_values.std(), 4)
        }
        profile[col_name] = stats
            
    return profile


if __name__ == "__main__":
    # --- 1. 參數設定 ---
    # 您可以透過修改這些參數，來建立各種不同的模型
    
    # 目標投手 (這個名稱會去對應 'pitch_record' 表中的 'player_name' 欄位)
    TARGET_PITCHER_NAME = "Ohtani, Shohei"
    
    # 目標球種 (這個名稱會去對應 'pitch_record' 表中的 'pitch_type' 欄位)
    # 如果設定為 None，則代表撈取該投手「所有球種」的好球
    TARGET_PITCH_TYPE = "FS"  # 例如: "FF", 或 None
    
    # 根據上面的參數，自動產生一個清晰、獨一無二的模型名稱
    if TARGET_PITCH_TYPE:
        MODEL_NAME = f"{TARGET_PITCHER_NAME}_{TARGET_PITCH_TYPE}_v1"
    else:
        MODEL_NAME = f"{TARGET_PITCHER_NAME}_all_v1"
    
    db = SessionLocal()
    try:
        # --- 2. 根據設定的條件，從資料庫撈取訓練資料 ---
        print(f"準備建立模型: {MODEL_NAME}")
        print(f"撈取條件 -> 投手: {TARGET_PITCHER_NAME}, 球種: {TARGET_PITCH_TYPE or '所有'}")

        # 建立一個查詢條件列表
        query_conditions = [
            PitchRecording.player_name == TARGET_PITCHER_NAME,
            # 篩選條件：description 欄位包含 'strike' 或 'foul' (ilike 不分大小寫)
            or_(
                PitchRecording.description.ilike('%strike%'),
                PitchRecording.description.ilike('%foul%')
            )
        ]
        
        # 如果有指定球種，就加入到查詢條件中
        if TARGET_PITCH_TYPE:
            query_conditions.append(PitchRecording.pitch_type == TARGET_PITCH_TYPE)
        
        # 執行查詢
        good_pitches_query = db.query(Kinematics).join(PitchRecording).filter(and_(*query_conditions))
        
        features_list = [p.__dict__ for p in good_pitches_query.all()]

        if not features_list:
            print("❌ 錯誤：在資料庫中找不到符合條件的訓練資料，無法建立模型。")
            exit()
            
        print(f"成功撈取 {len(features_list)} 筆訓練資料。")

        # --- 3. 建立統計模型 ---
        print(f"正在建立統計模型...")
        profile_dict = create_pitch_profile(features_list)

        if not profile_dict:
            print("❌ 錯誤：建立模型失敗，可能因為數據不足。")
            exit()

        # --- 4. 儲存模型到資料庫 (如果已存在則更新，不存在則新增) ---
        existing_profile = db.query(PitchModel).filter(PitchModel.model_name == MODEL_NAME).first()
        
        if existing_profile:
            print(f"模型 '{MODEL_NAME}' 已存在，將進行更新...")
            existing_profile.profile_data = profile_dict
            existing_profile.source_feature_count = len(features_list)
        else:
            print(f"正在將新模型 '{MODEL_NAME}' 存入資料庫...")
            new_profile = PitchModel(
                model_name=MODEL_NAME,
                profile_data=profile_dict,
                source_feature_count=len(features_list)
            )
            db.add(new_profile)
        
        db.commit()
        print(f"✅ 模型 '{MODEL_NAME}' 已成功儲存至資料庫！")

    except Exception as e:
        print(f"\n處理過程中發生嚴重錯誤: {e}")
        db.rollback()
    finally:
        db.close()