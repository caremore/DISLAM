import torch
import torch.nn as nn


class ModalityFusionTransformer(nn.Module):
    def __init__(self, image_dim=512, imu_dim=512, fusion_dim=512,
                 nhead=4, num_layers=4, dropout=0.1, max_seq_len=256):
        super().__init__()

        # 1. 将 image / imu 映射到统一维度
        # 2. 模态类型嵌入（图像=0，IMU=1）
        self.modality_embed = nn.Embedding(2, fusion_dim)

        # 3. 可学习的位置编码（基于 token 序号）
        self.position_embed = nn.Embedding(max_seq_len, fusion_dim)

        # 4. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=nhead,
            dim_feedforward=4 * fusion_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, image_feats, imu_feats):
        """
        :param image_feat: 图像特征 (B, T1, D1)
        :param imu_feat:   IMU特征 (B, T2, D2)

        :return: 编码后的融合特征 (B, T1+T2, fusion_dim)
        """

        image_last_features = [f[-1] for f in image_feats]
        vis_feat = torch.cat(image_last_features, 1)#[3, 512, 6, 20]
        B, C, H, W = vis_feat.shape
        vis_tokens = vis_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Step 2: 拼接模态
        fused_tokens = torch.cat([vis_tokens, imu_feats], dim=1)               # [3, 130, 512]
        B, T2, _ = fused_tokens.shape

        # # Step 3: 构造模态类型 id
        modality_ids = torch.cat([
            torch.zeros((B, vis_tokens.shape[1]), dtype=torch.long, device=vis_tokens.device),
            torch.ones((B, imu_feats.shape[1]), dtype=torch.long, device=imu_feats.device)
        ], dim=1)  # shape: (B, 2T) [3, 130]

        modality_encoding = self.modality_embed(modality_ids)  # (B, T1+T2 D)
        # # 4. 构造 token 位置索引：0 到 T1+T2-1
        position_ids = torch.arange(0, vis_tokens.shape[1] + imu_feats.shape[1],
                                    device=vis_tokens.device).unsqueeze(0).repeat(B, 1)  # (B, T1+T2)
        position_encoding = self.position_embed(position_ids)  # (B, 2T, D)

        # 5. 特征增强 + Transformer
        enhanced = fused_tokens + modality_encoding + position_encoding  # (B,  T1+T2, D)
        out = self.transformer_encoder(enhanced)  # (B, 2T, D)torch.Size([3, 130, 512])

        return out
