import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)
import pandas as pd

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
# def imutxt_loader(imu_path):
#     try:
#         with open(imu_path, 'rb') as f:  # 使用二进制模式确保线程安全
#             imu_data = pd.read_csv(f, header=None, names=[
#                 'timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'],
#                                    comment='#')
#             return imu_data
#     except Exception as e:
#         print(f"加载IMU数据失败: {e}")
#         return None
def imutxt_loader(imu_path):

    imu_data = []
    with open(imu_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # 跳过注释和空行
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            try:
                imu_data.append([
                    float(parts[0]),  # timestamp
                    float(parts[1]),  # wx
                    float(parts[2]),  # wy
                    float(parts[3]),  # wz
                    float(parts[4]),  # ax
                    float(parts[5]),  # ay
                    float(parts[6]),  # az
                ])
            except ValueError:
                continue
    return imu_data  # 返回 list[list[float]]


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        use_depth_hints
        depth_hint_path
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height = 192,
                 width = 640,
                 frame_idxs = [0,-1,1] ,
                 num_scales = 4,
                 is_train=False,
                 is_test = False):
        super(MonoDataset, self).__init__()

        # root_dir, seq_length = 3, imu_window = 0.2, transform = None, Hsize = 192, Wsize = 640, num_scales = 4,
        # split = 'train',

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        # self.interp = Image.ANTIALIAS
        self.interp = transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.imu_loader = imutxt_loader

        # We need to specify augmentations differently in newer versions of torchvision.
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1


        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)
        self.load_depth = True
        self.is_test = is_test
        self.load_imu = True


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_aug" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if color_aug is None:
                if "color" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if "color_aug" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
            else:
                if "color" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    if inputs[(n, im, i)].sum() == 0:
                        inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                    else:
                        inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))




    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps
            "depth_hint"                            for depth hint
            "depth_hint_mask"                       for mask of valid depth hints

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        seq, *timestamps = self.filenames[index]

        center_idx = len(timestamps) // 2
        center_ts = timestamps[center_idx]

        for i, ts in enumerate(timestamps):# -1,0,1 inputs[(items, i - center_idx, scale)] = tensor
            try:
                inputs[("color", i-1, -1)] = self.get_color(seq, ts, do_flip,is_seaErra = True)# image name + 0/-1/1
                inputs[("color_aug", i-1, -1)] = self.get_color(seq, ts, do_flip, do_flip)  # image name + 0/-1/1
            except FileNotFoundError as e:
                print(f"加载图像失败 ")

                inputs[("color", i-1, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))

                inputs[("color_aug", i-1, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            if seq in [ 'flatiron','u_canyon', 'tiny_canyon','horse_canyon']:#'flatiron','landward_path'
                K = self.K_canyons.copy()
            else:
                K = self.K_red_sea.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug or self.is_test:

            # color_aug = (lambda x: x)
            color_aug = None
        else:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]



        if self.load_depth:
            depth_gt = self.get_depth(seq, center_ts, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        # if self.is_test:
        #      inputs["folder"] = seq
        #      inputs["frame_index"] = center_ts
        if self.load_imu:
            inputs['imu'] = self._load_imu_segment(seq, center_ts)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    def _load_imu_segment(self, seq='tiny_canyon', target_time_ns=0, total_samples=20):
        raise NotImplementedError