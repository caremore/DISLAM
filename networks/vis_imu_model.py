import torch
from torch import nn
#from Data import  data
import numpy as np
from .multimodal_embedding import MultimodalEmbedding
from .multimodal_transformer import Mul_Encoder
class My_T_MAAM(nn.Module):
    def __init__(self,config):
        super(My_T_MAAM,self).__init__()
        self.config = config
        self.embedding = MultimodalEmbedding(config)
        self.transformer = Mul_Encoder(config)
        self.dense = nn.Linear(512, 512)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(512*3,512)
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512,1)
        self.jihuo = nn.Tanh()
        self.dropout = nn.Dropout(p=0.11)



    def forward(self , my_mask, image_feats , imu_embedding):
        image_last_features = [f[-1] for f in image_feats]
        vis_feat = torch.cat(image_last_features, 1)  # [3, 512, 6, 20]
        B, C, H, W = vis_feat.shape
        visual_embedding = vis_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        embedding_output = self.embedding(visual_embedding , imu_embedding)

        transformer_output = self.transformer(embedding_output,my_mask)[0]

        transformer_output_wo_esp = transformer_output[:, 1:self.config['vis_len'], :]  # 视觉部分
        imu_output_wo_esp = transformer_output[:, self.config['vis_len'] + 1:, :]  # IMU部分
        fused_valid = torch.cat([transformer_output_wo_esp, imu_output_wo_esp], dim=1)  # [B, T_valid, D]

        return fused_valid


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.kaiming_uniform_(m.weight)


def save_model(model,acc,epoch):

    print("Updating the model")
    model_dict = {
        'state_dict': model.state_dict(),
        'acc': acc,
    }
    path = './weights/' + str(epoch)+'model.pt'
    torch.save(model_dict, path)





if __name__ == '__main__':
    # 参数设定
    B = 4  # batch size
    vis_len = 120  # 视觉 token 数（不含 cls）
    imu_len = 10  # IMU token 数（不含 cls）
    hidden_size = 512  # 每个 token 的特征维度

    # 模拟视觉特征：比如来自 CNN 的 patch 特征
    visual_input = torch.randn(B, vis_len, hidden_size)  # 去掉1，是因为esp_1要加进来
    imu_input = torch.randn(B, imu_len, hidden_size)  # 同理















