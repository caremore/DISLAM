# import torch
# import torch.nn as nn
#
#
# class PoseDecoderBiLSTM(nn.Module):
#     def __init__(self, input_dim=512, hidden_dim=256, num_layers=1):
#         super().__init__()
#         self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
#                               bidirectional=True, batch_first=True)
#         self.fc_rot = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # 输出 axis-angle（旋转向量）
#         )
#
#         self.fc_trans = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # 输出平移向量
#         )
#
#     def forward(self, encoded_seq):
#         """
#         :param encoded_seq: Transformer 编码器输出 (B, T, D)
#         :return: 旋转（axis-angle）和平移向量，各为 (B, 3)
#         """
#         lstm_out, _ = self.bilstm(encoded_seq)  # (B, T, 2*hidden_dim)
#         last_feat = lstm_out[:, -1, :]  # 取最后时间步的表示 (B, 2*hidden_dim)
#
#         axisangle = self.fc_rot(last_feat).view(-1, 1, 1, 3)  # (B, 3)
#         translation = self.fc_trans(last_feat).view(-1, 1, 1, 3)  # (B, 3)
#
#         return axisangle, translation
import torch
import torch.nn as nn

class PoseDecoderBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, num_frames_to_predict_for=2):
        super().__init__()
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True)

        self.pose_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 6 * num_frames_to_predict_for)
        )

        #self.learnable_scale = nn.Parameter(torch.tensor(0.01))  # 初始值设为 0.01，可调

    def forward(self, encoded_seq):
        """
        :param encoded_seq: Transformer 编码器输出 (B, T, D)
        :return: (axisangle, translation): shape = [B, T, 1, 3]
        """
        lstm_out, _ = self.bilstm(encoded_seq)  # [B, T, 2H]
        last_feat = lstm_out[:, -1, :]          # [B, 2H]

        pose = self.pose_fc(last_feat)          # [B, 6 * T]
        pose = 0.01 * pose.view(-1, self.num_frames_to_predict_for, 6)  # [B, T, 6]

        axisangle = pose[..., :3].unsqueeze(2)  # [B, T, 1, 3]
        #translation = self.learnable_scale * torch.tanh(pose[..., 3:].unsqueeze(2))  # [B, T, 1, 3]
        translation = pose[..., 3:].unsqueeze(2)  # [B, T, 1, 3]

        return axisangle, translation
