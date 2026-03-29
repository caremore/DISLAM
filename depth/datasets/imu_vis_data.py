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
import os
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ColorJitter

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
class VisualInertialDataset(Dataset):
    def __init__(self, root_dir, seq_length=3, imu_window=0.2, transform=None,Hsize=192,Wsize=640,num_scales=4,
                 split='train', val_size=0.1, test_size=0.1, random_state=42,is_test=False):
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
        self.is_test=is_test

        self.jitter_fn = transforms.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=(-0.1, 0.1))


    def _scan_sequences(self):
        """扫描所有子目录（每个序列）"""
        return [d.name for d in self.root.iterdir() if d.is_dir()]

    # def random_jitter(self):
    #     return transforms.ColorJitter(
    #         brightness=(0.8, 1.2),
    #         contrast=(0.8, 1.2),
    #         saturation=(0.8, 1.2),
    #         hue=(-0.1, 0.1)
    #     )
    def _build_frame_pairs(self):
        """构建所有序列中有效的连续多帧图像样本"""
        pairs = []
        for seq in self.sequences:
            frame_paths = sorted((self.root / seq / 'imgs').glob('*.tiff'))
            if len(frame_paths) < self.seq_length:
                continue
            timestamps = [int(f.stem) for f in frame_paths]
            for i in range(len(timestamps) - self.seq_length + 1):
                selected = timestamps[i:i + self.seq_length]
                pairs.append((seq, *selected))
        return pairs

    def _prepare_splits(self, val_size=0.1, test_size=0.1):
        """划分训练、验证、测试集，支持保存和加载划分结果"""
        split_file =  self.root / f'splits_{self.seq_length}f.json'

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

    def load_multiscale_images(self, timestamps, seq, center_idx, kind_of_image, items):
        inputs = {}

        timestamps = sorted(timestamps)  # 保证时间顺序 t-1, t, t+1
        for i, ts in enumerate(timestamps):
            # 构建路径
            img_dir = self.root / seq / kind_of_image
            suffix = "_SeaErra.tiff" if kind_of_image == 'seaErra' else ".tiff"
            img_path = img_dir / f"{ts}{suffix}"
            # 安全加载图像（三层防护）
            try:
                with open(img_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                    # 3. 多尺度处理
                        for scale in range(self.num_scales):
                            with torch.no_grad():  # 避免梯度跟踪
                                resized = self.resize[scale](img)
                                tensor = self.transform(resized)
                                offset = i - center_idx  # 正确偏移
                                inputs[(items, offset, scale)] = tensor

            except Exception as e:
                print(f"加载图像失败 {img_path}: {e}")
                # 提供降级方案
                dummy = torch.zeros(3, self.height, self.width)
                for scale in range(self.num_scales):
                    inputs[(items, i - center_idx, scale)] = dummy.clone()

        # ✅ 拼接 color_aug 的三帧为横图（仅 scale=0）

        # import torchvision.utils as vutils
        #
        # if items == "color_aug":
        #     triplet = []
        #     for offset in [-1, 0, 1]:
        #         key = ("color_aug", offset, 0)
        #         if key in inputs:
        #             triplet.append(inputs[key])
        #
        #     if len(triplet) == 3 and len(SS) >= 3:
        #         os.makedirs("logs/debug_triplet", exist_ok=True)
        #
        #         # 1. 拼接 transform 后的图像 (tensor)
        #         triplet_tensor = torch.stack(triplet, dim=0)  # [3, C, H, W]
        #
        #         # 2. 拼接原始 PIL 图像并转为 tensor
        #         pil_triplet = SS[:3]
        #         pil_triplet_resized = [img.resize((self.width, self.height)) for img in pil_triplet]
        #         pil_tensor = torch.stack([transforms.ToTensor()(img) for img in pil_triplet_resized])  # [3, C, H, W]
        #
        #         # 3. 拼接成 2 行 3 列图像网格
        #         all_6 = torch.cat([pil_tensor, triplet_tensor], dim=0)  # [6, C, H, W]
        #         grid = vutils.make_grid(all_6, nrow=3, padding=2)  # 两行三列
        #
        #         # 4. 自动命名保存文件，使用 timestamp 和 center_idx 避免覆盖
        #         ts_str = "_".join([str(t) for t in timestamps])
        #         vutils.save_image(grid, f"logs/debug_triplet/grid_2x3_center{center_idx}_{ts_str}.png")

        return inputs

    def load_depth(self,seq,ts):
        inputs = {}
        depth_path =  self.root / seq / 'depth' / f'{ts}_SeaErra_abs_depth.tif'
        with pil.open(depth_path) as depth_img:
            depth_gt = np.array(depth_img).astype(np.float32)

        depth_gt = np.expand_dims(depth_gt, 0)
        depth_tensor = torch.from_numpy(depth_gt)

        inputs['depth_gt'] = depth_tensor

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

    def _load_imu_segment(self, seq='tiny_canyon', target_time_ns=0, total_samples=20):
        """
        根据纳秒级中心时间戳提取IMU数据窗口，并统一长度为20。
        返回格式为 np.ndarray: [20, 7]，即 [timestamp, wx, wy, wz, ax, ay, az]
        """
        inputs = {}

        imu_file = 'IMU_interp.txt' if seq in ['flatiron', 'horse_canyon', 'tiny_canyon', 'u_canyon'] else 'imu.txt'
        imu_path = self.root / seq / imu_file

        half = total_samples // 2


        try:
            with open(imu_path, 'rb') as f:  # 使用二进制模式确保线程安全
                imu_data = pd.read_csv(f, header=None, names=[
                    'timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'],
                                       comment='#')
        except Exception as e:
            print(f"加载IMU数据失败: {e}")
            return None

        timestamps = imu_data['timestamp'].values

        target_ts_ns = target_time_ns / 1e7
        #
        # # 找到最接近目标时间的索引
        closest_idx = np.abs(timestamps - target_ts_ns).argmin()

        start_idx = max(0, closest_idx - half)
        end_idx = min(len(imu_data), start_idx + total_samples)

        segment = imu_data.iloc[start_idx:end_idx].reset_index(drop=True)

        if len(segment) < total_samples:
            print(f"[警告] IMU 数据不足 {total_samples} 条，当前为 {len(segment)} 条，尝试从次接近时间戳补齐")
            return torch.zeros((20, 6), dtype=torch.float32)

        #return torch.tensor(segment[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].values, dtype=torch.float32)

        imu_tensor = torch.from_numpy(segment[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].values)
        return imu_tensor.float()  # 在此处转换类型

    def build_my_mask_only_vis_imu(self,vis_len, imu_len):
        """
        构建视觉和IMU模态的 my_mask，考虑了前置的 esp token。

        参数:
            VIS_n: 有效视觉帧数（不含esp）
            IMU_n: 有效IMU帧数（不含esp）
        返回:
            my_mask: shape=(1, max_vlen + max_alen)
        """
        mask_vis = np.ones(vis_len + 1)  # 视觉部分 + esp
        mask_imu = np.ones(imu_len + 1)  # IMU部分 + esp
        my_mask = np.concatenate([mask_vis, mask_imu], axis=0)
        # return my_mask.reshape(1, -1)
        return torch.tensor(my_mask, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):

        inputs = {}

        do_color_aug = self.is_train and random.random() >= 0.5
        do_flip = self.is_train and random.random() > 0.5
        #jitter_fn = self.random_jitter()

        seq, *timestamps = self.frame_pairs[idx]
        center_idx = len(timestamps) // 2
        center_ts = timestamps[center_idx]

        #加载图像序列并生成多尺度图像 def load multi-scale images
        out_color_images = self.load_multiscale_images(timestamps, seq, center_idx, 'seaErra', 'color')
        inputs.update(out_color_images)
        # if do_color_aug:
        #     out_color_aug_images = self.load_multiscale_images(timestamps, seq, center_idx, 'imgs', 'color_aug')
        #     # inputs.update(out_color_aug_images)
        # else:
        #     out_color_aug_images = self.load_multiscale_images(timestamps, seq, center_idx, 'seaErra', 'color_aug')
        # inputs.update(out_color_aug_images)

        if do_color_aug or self.is_test:
            out_color_aug_images = self.load_multiscale_images(timestamps, seq, center_idx, 'imgs', 'color_aug')
            inputs.update(out_color_aug_images)
        else:
            for k in list(inputs):
                f = inputs[k]
                if k[0] == "color":
                    _, im, s = k
                    inputs[("color_aug", im, s)] = self.jitter_fn(f.clone())

        # 加载深度图def load_depth
        inputs.update(self.load_depth(seq, center_ts))

        #加载多尺度内参
        inputs.update(self.load_and_scale_intrinsics(seq, 4))


        # 添加时间戳信息
        inputs['timestamps'] = torch.tensor(timestamps)

        #加载 IMU 序列
        inputs['imu'] = self._load_imu_segment(seq, center_ts)


        inputs['my_mask'] = self.build_my_mask_only_vis_imu(144, 10)#120 256 144


        return inputs

    def __len__(self):
        return len(self.frame_pairs)


# 使用示例：
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = VisualInertialDataset("/root/autodl-fs/pythondata/FLsea", split='train')
    print(f"数据集包含 {len(dataset)} 个训练样本")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False,
                                     num_workers=8,
                                     pin_memory=False, drop_last=True)
    i = 0
    for batch_idx, inputs in enumerate(train_loader):
           print(batch_idx)