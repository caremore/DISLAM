
import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from .mono_dataset import  MonoDataset
import cv2
import pandas as pd
import torch

class FSEADataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(FSEADataset, self).__init__(*args, **kwargs)
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation. 128 416
        RAW_WIDTH = 968
        RAW_HEIGHT = 608

        self.K_canyons = np.array([[1175.3913, 0,  466.2595, 0],
                                   [0, 1174.2805, 271.2117, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)
        self.K_red_sea = np.array([[1296.6668, 0, 501.5039, 0],
                                   [0, 1300.8313, 276.1617, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)

        self.K_canyons[0, :] /=  RAW_WIDTH
        self.K_canyons[1, :] /=  RAW_HEIGHT

        self.K_red_sea[0, :] /= RAW_WIDTH
        self.K_red_sea[1, :] /= RAW_HEIGHT





    def get_color(self, seq, i, do_flip,is_seaErra = False):
        if is_seaErra:
            color = self.loader(self.get_seaErraimage_path(seq, i))
        else:
            color = self.loader(self.get_image_path(seq, i))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class FSEADepthDataset(FSEADataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(FSEADepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        f_str = "{}.tiff".format(frame_index)
        image_path = os.path.join(
            self.data_path,
            folder,
            "imgs",#seaErra
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, do_flip):
        f_str = "{}{}.tif".format(frame_index,'_SeaErra_abs_depth')
        depth_path = os.path.join(
            self.data_path,
            folder,
            "depth",
            f_str)
        depth_gt = pil.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_seaErraimage_path(self, folder, frame_index):
        f_str = "{}{}.tiff".format(frame_index,'_SeaErra')
        image_path = os.path.join(
            self.data_path,
            folder,
            "seaErra",#seaErra
            f_str)
        return image_path

    def _load_imu_segment(self, seq='tiny_canyon', target_time_ns=0, total_samples=20):
        imu_file = 'IMU_interp.txt' if seq in ['flatiron', 'horse_canyon', 'tiny_canyon', 'u_canyon'] else 'imu.txt'
        imu_path = os.path.join(self.data_path, seq, imu_file)

        half = total_samples // 2

        imu_data = self.imu_loader(imu_path)
        if not imu_data:
            print("IMU 数据为空")
            return torch.zeros((20, 6), dtype=torch.float32)

        imu_data = sorted(imu_data, key=lambda x: x[0])  # 按 timestamp 排序
        timestamps = [row[0] for row in imu_data]

        target_ts_sec = target_time_ns / 1e7
        closest_idx = np.argmin(np.abs(np.array(timestamps) - target_ts_sec))

        start_idx = max(0, closest_idx - half)
        end_idx = min(len(imu_data), start_idx + total_samples)
        segment = imu_data[start_idx:end_idx]

        if len(segment) < total_samples:
            print(f"[警告] IMU 数据不足 {total_samples} 条，当前为 {len(segment)} 条")
            return torch.zeros((20, 6), dtype=torch.float32)

        imu_values = [row[1:] for row in segment]  # 只保留 wx...az
        return torch.tensor(imu_values, dtype=torch.float32)

