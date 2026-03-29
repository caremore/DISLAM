import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json
import random
import numpy as np
import PIL.Image as pil
import logging
import matplotlib.pyplot as plt

# def plot_image_sequence(sample, figsize=(12, 4)):
#     images = sample['images']         # Tensor: [N, C, H, W]
#     timestamps = sample['timestamps'] # Tensor: [N]
#
#     N = images.shape[0]
#     print(images.shape)
#     plt.figure(figsize=figsize)
#
#     for i in range(N):
#         img = images[i].permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]
#         plt.subplot(1, N, i + 1)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(f"{timestamps[i].item()}", fontsize=8)
#
#     plt.tight_layout()
#     plt.show()
def load_depth(seq, ts):
    inputs = {}
    depth_path = Path('/root/autodl-fs/pythondata/FLsea') / seq / 'depth' / f'{ts}_SeaErra_abs_depth.tif'
    depth_gt = pil.open(depth_path)
    depth_gt = np.array(depth_gt).astype(np.float32)

    depth_tensor = torch.from_numpy(depth_gt)

    #inputs['depth_gt'] = depth_tensor

    return depth_tensor

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
    # from datetime import datetime
    # print(  datetime.utcfromtimestamp(segment['timestamp'].values[0]))
    # print(datetime.utcfromtimestamp(target_ts_ns))

    if len(segment) < total_samples:
        print(f"[警告] IMU 数据不足 {total_samples} 条，当前为 {len(segment)} 条，尝试从次接近时间戳补齐")
        return 0

    return 1
class VisualInertialDataset(Dataset):
    def __init__(self, root_dir, seq_length=3, imu_window=0.2, transform=None,Hsize=192,Wsize=640,num_scales=4,
                 split='train', val_size=0.1, test_size=0.1, random_state=42):
        """
        :param root_dir: 数据集根目录
        :param seq_length: 图像序列长度 (默认3帧)
        :param imu_window: IMU数据时间窗口 (秒)
        :param transform: 数据增强（如 ToTensor + Resize 等）
        :param split: 'train' | 'val' | 'test'
        :param val_size: 验证集比例
        :param test_size: 测试集比例
        """
        #tf = transforms.Compose([transforms.Resize((Hsize, Wsize)), transforms.ToTensor()])
        self.root = Path(root_dir)
        self.seq_length = seq_length
        self.imu_window = imu_window
        self.transform = transform or transforms.ToTensor()
        self.split = split
        self.random_state = random_state
        self.height = Hsize
        self.width = Wsize
        self.num_scales = num_scales

        self.is_train = split == 'train'

        self.interp = transforms.InterpolationMode.LANCZOS

        self.K_canyons = np.array([[1.21, 0, 0.48, 0],
                                   [0, 1.93, 0.44, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)
        self.K_red_sea = np.array([[1.34, 0, 0.52, 0],
                                   [0, 2.14, 0.45, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)

        # 初始化序列列表
        self.sequences = self._scan_sequences()

        # 自动划分数据集
        self._prepare_splits(val_size=val_size, test_size=test_size)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)

    def _scan_sequences(self):
        """扫描所有子目录（每个序列）"""
        return [d.name for d in self.root.iterdir() if d.is_dir()]

    def _build_frame_pairs(self):
        """构建所有序列中有效的连续多帧图像样本"""
        pairs = []
        for seq in self.sequences:
            if seq == 'pier_path':
                continue
            frame_paths = sorted((self.root / seq / 'imgs').glob('*.tiff'))
            if len(frame_paths) < self.seq_length:
                continue
            timestamps = [int(f.stem) for f in frame_paths]
            for i in range(len(timestamps) - self.seq_length + 1):
                selected = timestamps[i:i + self.seq_length]
                target_time_ns = int(selected[1])
                s = _load_imu_segment(seq = seq , target_time_ns = target_time_ns)
                _depth = load_depth(seq = seq, ts = target_time_ns)
                b = _depth.max()
                if s == 0 or b == 0:
                    continue
                else:
                    pairs.append((seq, *selected))
        return pairs

    def _prepare_splits(self, val_size=0.1, test_size=0.1):
        """划分训练、验证、测试集，支持保存和加载划分结果"""
        split_file = self.root / f'splits_{self.seq_length}f.json'

        if split_file.exists():
            print(f"读取已有划分文件: {split_file}")
            with open(split_file, 'r') as f:
                splits = json.load(f)
        else:
            print("第一次划分数据集，正在保存划分结果...")
            all_pairs = self._build_frame_pairs()
            remaining, test_set = train_test_split(all_pairs, test_size=test_size, random_state=self.random_state)
            train_set, val_set = train_test_split(remaining, test_size=val_size / (1 - test_size),
                                                  random_state=self.random_state)
            splits = {
                'train': train_set,
                'val': val_set,
                'test': test_set
            }
            for k in splits:
                splits[k] = [list(item) for item in splits[k]]
            with open(split_file, 'w') as f:
                json.dump(splits, f, indent=2)

        self.frame_pairs = [tuple(x) for x in splits[self.split]]
        logging.info(f"{self.split} 集加载成功，数量: {len(self.frame_pairs)}")

    def load_multiscale_images(self,timestamps,seq,center_idx,kind_of_image,items):
        inputs = {}
        for i, ts in enumerate(timestamps):
            if kind_of_image == 'seaErra':
                img_path = self.root / seq / kind_of_image / f'{ts}_SeaErra.tiff'
            else:
                img_path = self.root / seq / kind_of_image / f'{ts}.tiff'
            img = Image.open(img_path).convert('RGB')
            for scale in range(self.num_scales):
                resized = self.resize[scale](img)
                tensor = self.transform(resized)
                inputs[items, i - center_idx, scale] = tensor
        return inputs

    def load_and_scale_intrinsics(self,seq,scales=4):
        """加载标定文件并生成多尺度内参及逆矩阵"""
        if seq in ['flatiron','horse_canyon','tiny_canyon','u_canyon']:
            K_original =  self.K_canyons.copy()
        else:
            K_original = self.K_red_sea.copy()

        # 2. 为每个尺度生成内参和逆矩阵
        inputs = {}
        for scale in range(scales):
            # 缩放内参
            K_scaled = K_original.copy()
            scale_x =  self.width // (2 ** scale)
            scale_y =  self.height // (2 ** scale)

            K_scaled[0, :] *= scale_x
            K_scaled[1, :] *= scale_y

            # 计算逆矩阵并转为PyTorch Tensor
            inv_K_scaled = np.linalg.inv(K_scaled)

            inputs[("K", scale)] = torch.from_numpy(K_scaled).float()
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K_scaled).float()

        return inputs

    def _compute_relative_pose(self, ts1, ts3):
        """TODO: 根据时间戳计算相对位姿标签（占位）"""
        return torch.eye(4)  # 假设单位矩阵，真实任务中应替换为有效标签

    def __getitem__(self, idx):

        inputs = {}

        do_color_aug = self.is_train and random.random() >= 0.5
        do_flip = self.is_train and random.random() > 0.5

        seq, *timestamps = self.frame_pairs[idx]
        center_idx = len(timestamps) // 2
        center_ts = timestamps[center_idx]

        #加载图像序列并生成多尺度图像 def load multi-scale images
        out_color_images = self.load_multiscale_images(timestamps, seq, center_idx, 'seaErra', 'color')
        inputs.update(out_color_images)
        if do_color_aug:
            out_color_aug_images = self.load_multiscale_images(timestamps, seq, center_idx, 'imgs', 'color_aug')
            inputs.update(out_color_aug_images)
        else:
            for k in list(inputs):
                if "color" in k[0]:  # Check image_type (first element of tuple key)
                    n, im, s = k
                    inputs[(f"{n}_aug", im, s)] = inputs[k].clone()

        # 加载深度图def load_depth
        inputs.update(self.load_depth(seq, center_ts))

        #加载多尺度内参
        inputs.update(self.load_and_scale_intrinsics(seq, 4))


        # 添加时间戳信息
        inputs['timestamps'] = torch.tensor(timestamps)

        #加载 IMU 序列
        # imu_segment = self._load_imu_segment(seq, center_ts)
        # if imu_segment is None:
        #     imu_tensor = torch.zeros((20, 6), dtype=torch.float32)
        # else:
        #     imu_tensor = torch.tensor(imu_segment[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].values, dtype=torch.float32)
        #
        # inputs['imu'] = imu_tensor

        return inputs

    def __len__(self):
        return len(self.frame_pairs)


# 使用示例：
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = VisualInertialDataset("/root/autodl-fs/pythondata/FLsea", split='train')
    print(f"数据集包含 {len(dataset)} 个训练样本")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                   num_workers=1,
                                   pin_memory=True, drop_last=True)
    for batch_idx, inputs in enumerate(train_loader):
        print(batch_idx)

    #plot_image_sequence(sample)
    # print(f"图像 shape: {sample['images'].shape}")
    # print(f"IMU shape: {sample['imu'].shape}")
    # print(f"时间戳: {sample['timestamps']}")