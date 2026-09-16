import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# 전처리 모듈 임포트
from preprocessing import extract_advanced_features_v3

import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature

# 1. 데이터 로드
if os.path.exists('/content/drive/MyDrive'):
    FINAL_WORK_DIR = os.getenv('DATA_DIR', '/content/drive/MyDrive/canopy/od_gps_processed')
else:
    FINAL_WORK_DIR = os.getenv('DATA_DIR', './your_data_path_here')

print("Parquet 데이터 로드 중...")
df_train_full = pd.read_parquet(os.path.join(FINAL_WORK_DIR, 'train_10000.parquet'))
df_val_full = pd.read_parquet(os.path.join(FINAL_WORK_DIR, 'val_3000.parquet'))

train_trip_ids = df_train_full['trip_id'].unique()
np.random.seed(42)
np.random.shuffle(train_trip_ids)
selected_train_ids = train_trip_ids[:1000]
train_df = df_train_full[df_train_full['trip_id'].isin(selected_train_ids)].copy()

val_trip_ids = df_val_full['trip_id'].unique()
np.random.shuffle(val_trip_ids)
selected_val_ids = val_trip_ids[:400]
df_val_subset = df_val_full[df_val_full['trip_id'].isin(selected_val_ids)].copy()

val_ids = selected_val_ids[:200]
test_ids = selected_val_ids[200:]

val_df = df_val_subset[df_val_subset['trip_id'].isin(val_ids)].copy()
test_df = df_val_subset[df_val_subset['trip_id'].isin(test_ids)].copy()

del df_train_full, df_val_full, df_val_subset

for df in [train_df, val_df, test_df]:
    if 'speed_ms' in df.columns and 'speed' not in df.columns:
        df['speed'] = df['speed_ms']

# 라벨 인덱스 변환
TRAIN_TO_COMPACT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
def apply_compact_labels(df):
    df = df.copy()
    if 'mode' in df.columns:
        df['mode_compact'] = df['mode'].map(TRAIN_TO_COMPACT)
    return df

train_df = apply_compact_labels(train_df)
val_df = apply_compact_labels(val_df)
test_df = apply_compact_labels(test_df)

# 2. 전처리 함수 적용
print("피처 추출 진행 중 (Train, Val, Test)...")
train_df = extract_advanced_features_v3(train_df)
val_df = extract_advanced_features_v3(val_df)
test_df = extract_advanced_features_v3(test_df)

# 3. 모델 학습
features = [
    'speed', 'acceleration', 'distance', 'bearing_change',
    'speed_mean_5', 'speed_std_5', 'speed_max_10',
    'speed_mean_30', 'speed_std_30', 'speed_max_60', 'speed_mean_150',
    'stop_count_60', 'stop_count_150', 'stoppage_ratio_60',
    'speed_q25_60', 'speed_q75_60', 'accel_std_30'
]

X_train, y_train = train_df[features], train_df['mode_compact']
X_val, y_val = val_df[features], val_df['mode_compact']

with mlflow.start_run() as run:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "random_state": 42
    }
    mlflow.log_params(params)

    print("LightGBM 모델 학습 시작")
    model = lgb.LGBMClassifier(**params, n_jobs=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # 간단 평가
    preds = model.predict(X_val[features])
    acc = accuracy_score(y_val, preds)
    mlflow.log_metric("val_accuracy", acc)

    # 4. MLflow 등록
    input_example = X_val.head(5)
    signature = infer_signature(input_example, model.predict(input_example))
    registry_name = "dbw_canopy_dev.ml.canopy_transition_lgbm_model"
    
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registry_name
    )
    print(f"MLflow 모델이 [{registry_name}] 스키마에 성공적으로 등록되었습니다")
