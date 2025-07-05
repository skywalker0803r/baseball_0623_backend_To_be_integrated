import pandas as pd
import numpy as np

def classify_ball_quality(ball_json, model, target_length=239):
    """
    這個函數ball_json就是棒球api回傳的json檔案
    model就是一個隨機森林模型
    輸出浮點數代表是好球的機率
    """
    results = ball_json['results']
    x_list = []
    y_list = []

    for item in results:
        coords = item[1]
        # Check if coords is not None and has 4 elements
        if coords is not None and len(coords) == 4:
            x1, y1, x2, y2 = coords
            # Ensure all coordinates are not None. If any are, append None.
            if all(c is not None for c in [x1, y1, x2, y2]):
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                x_list.append(center_x)
                y_list.append(center_y)
            else:
                x_list.append(None)
                y_list.append(None)
        else:
            # If coords itself is None or malformed, append None
            x_list.append(None)
            y_list.append(None)

    # Pad with None values until target_length
    while len(x_list) < target_length:
        x_list.append(None)
        y_list.append(None)

    # Truncate to target_length (as a safeguard)
    x_list = x_list[:target_length]
    y_list = y_list[:target_length]

    # Create column names
    columns = [f'x_{i}' for i in range(target_length)] + [f'y_{i}' for i in range(target_length)]
    values = x_list + y_list
    
    # Create DataFrame. Pandas can handle None, which will become NaN.
    df = pd.DataFrame([values], columns=columns)

    # Note: Many machine learning models, including RandomForest, do not natively
    # handle NaN values. You might need to impute or handle these NaNs before prediction.
    # For demonstration, I'm leaving it as is, assuming your model or pipeline
    # is set up to handle potential NaNs (which Pandas will convert from None).
    
    return float(model.predict_proba(df)[0][0])