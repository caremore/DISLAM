
import pandas as pd
import numpy as np
from pathlib import Path


def _load_imu_segment(seq='tiny_canyon', target_time_ns = 0, total_samples=20):
    """
    根据纳秒级中心时间戳提取IMU数据窗口，并统一长度为20。
    返回格式为 np.ndarray: [20, 7]，即 [timestamp, wx, wy, wz, ax, ay, az]
    """
    imu_file = 'IMU_interp.txt' if seq in ['flatiron', 'horse_canyon', 'tiny_canyon', 'u_canyon'] else 'imu.txt'
    imu_path = Path('/root/autodl-fs/pythondata/FLsea') / seq / imu_file

    half = total_samples // 2

    imu_data = pd.read_csv(imu_path, header=None, names=[
        'timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az',
    ], comment='#')
    # imu_data['timestamp'] = (imu_data['timestamp'].values * 1e9).astype(np.int64)
    timestamps = imu_data['timestamp'].values

    target_ts_ns = target_time_ns / 1e7
    #
    # # 找到最接近目标时间的索引
    closest_idx = np.abs(timestamps - target_ts_ns).argmin()

    start_idx = max(0, closest_idx - half)
    end_idx = min(len(imu_data), start_idx + total_samples)

    segment = imu_data.iloc[start_idx:end_idx].reset_index(drop=True)
    from datetime import datetime
    print(  datetime.utcfromtimestamp(segment['timestamp'].values[0]))
    print(datetime.utcfromtimestamp(target_ts_ns))

    if len(segment) < total_samples:
        print(f"[警告] IMU 数据不足 {total_samples} 条，当前为 {len(segment)} 条，尝试从次接近时间戳补齐")
        return 1

    return 0
if __name__ == "__main__":
    import os

    path = '/root/autodl-fs/pythondata/FLsea/tiny_canyon/imgs' #pier_path tiny_canyon
    i = 0
    a = 0
    for img in os.listdir(path):
        a= a+1
        filename = os.path.splitext(img)[0]  # 去掉后缀
        target_time_ns = int(filename)
        s = _load_imu_segment(target_time_ns = target_time_ns)

        i = i + s
        print(i)
    print(a)