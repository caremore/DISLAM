import torch
import torch.nn as nn

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, leaky_relu_slope=0.1, norm=True):
        super(IMUEncoder, self).__init__()
        self.do_norm = norm
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        )

    def forward(self, imu_data):
        """
        :param imu_data: 输入形状为 (B, T, 6)
        :return: 输出形状为 (B, T, 512)
        """
        if self.do_norm:
            accel_data = imu_data[:, :, :3]
            gyro_data = imu_data[:, :, 3:]

            # Min-Max归一化到[-1, 1]
            accel_min, _ = accel_data.min(dim=1, keepdim=True)
            accel_max, _ = accel_data.max(dim=1, keepdim=True)
            accel_range = accel_max - accel_min + 1e-5
            accel_normalized = 2 * (accel_data - accel_min) / accel_range - 1

            gyro_min, _ = gyro_data.min(dim=1, keepdim=True)
            gyro_max, _ = gyro_data.max(dim=1, keepdim=True)
            gyro_range = gyro_max - gyro_min + 1e-5
            gyro_normalized = 2 * (gyro_data - gyro_min) / gyro_range - 1

            imu_data = torch.cat([accel_normalized, gyro_normalized], dim=-1)

        x = imu_data.permute(0, 2, 1)  # (B, 6, T)
        x = self.encoder(x)           # (B, 512, T)
        x = x.permute(0, 2, 1)        # (B, T, 512)
        return x


# # 初始化
# imu_encoder = IMUEncoder(input_dim=6)
#
# # 模拟输入数据 (batch_size=2, N=10个IMU样本, 6维数据)
# imu_data = torch.randn(2, 10, 6)
#
# # 前向传播
# features = imu_encoder(imu_data)
# print(features.shape)  # 输出: torch.Size([2, 10, 64])