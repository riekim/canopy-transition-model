import numpy as np
import pandas as pd

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return np.degrees(np.arctan2(x, y))

def extract_advanced_features_v3(df):
    """
    이벤트 허브나 실시간 스트리밍, 학습 데이터 모두에서 공통으로 쓰이는 전처리 함수
    """
    processed_dfs = []

    for trip_id, group in df.groupby('trip_id'):
        group = group.reset_index(drop=True)
        lat = group['latitude'].values
        lon = group['longitude'].values
        ts = group['timestamp'].values.astype(np.float64)

        ts_sec = ts / 1000.0 if ts[0] > 1e11 else ts
        dt = np.clip(np.diff(ts_sec, prepend=ts_sec[0]), 0.1, None)

        dist = np.zeros(len(group))
        lat1, lon1, lat2, lon2 = map(np.radians, [lat[:-1], lon[:-1], lat[1:], lon[1:]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        dist[1:] = 6371000 * c

        speed = (dist / dt) * 3.6
        acceleration = np.diff(speed, prepend=speed[0]) / dt

        group['speed'] = speed
        group['acceleration'] = acceleration
        group['distance'] = dist

        bearings = np.zeros(len(group))
        if len(group) > 1:
            bearings[1:] = calculate_bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
        bearing_diff = np.abs(np.diff(bearings, prepend=bearings[0]))
        bearing_diff = np.where(bearing_diff > 180, 360 - bearing_diff, bearing_diff)
        group['bearing_change'] = bearing_diff

        speed_series = pd.Series(speed)
        accel_series = pd.Series(acceleration)

        group['speed_mean_5'] = speed_series.rolling(window=5, min_periods=1).mean()
        group['speed_std_5'] = speed_series.rolling(window=5, min_periods=1).std().fillna(0)
        group['speed_max_10'] = speed_series.rolling(window=10, min_periods=1).max()
        group['speed_mean_30'] = speed_series.rolling(window=30, min_periods=1).mean()
        group['speed_std_30'] = speed_series.rolling(window=30, min_periods=1).std().fillna(0)
        group['speed_max_60'] = speed_series.rolling(window=60, min_periods=1).max()
        group['speed_mean_150'] = speed_series.rolling(window=150, min_periods=1).mean()

        is_stopped = (speed < 3.0).astype(int)
        stopped_series = pd.Series(is_stopped)
        group['stop_count_60'] = stopped_series.rolling(window=60, min_periods=1).sum()
        group['stop_count_150'] = stopped_series.rolling(window=150, min_periods=1).sum()
        group['stoppage_ratio_60'] = stopped_series.rolling(window=60, min_periods=1).mean()

        group['speed_q25_60'] = speed_series.rolling(window=60, min_periods=1).quantile(0.25)
        group['speed_q75_60'] = speed_series.rolling(window=60, min_periods=1).quantile(0.75)
        group['accel_std_30'] = accel_series.rolling(window=30, min_periods=1).std().fillna(0)

        processed_dfs.append(group)

    return pd.concat(processed_dfs, ignore_index=True)
