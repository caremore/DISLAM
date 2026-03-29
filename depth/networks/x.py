import torch
from torch import nn
#from Data import  data
import numpy as np
import multimodal_embedding , multimodal_transformer
config = {
    "imu_len": 11,
    "vis_len": 121,
    "hidden_size": 512,
    #"max_position": 150,
    "modal_size": 2,
    "layer_norm_eps": 1e-5,
    "hidden_dropout_prob": 0.1,         # dropout rate
    "visual_dim": 120,                   # 原始视觉特征维度
    "imu_dim": 10,                     # 原始音频特征维度
    "attention_dropout_prob": 0.1,
    "num_head": 4,
    "output_attention": 0,
    "num_layer": 3,
    "output_hidden_state": 0,
    "intermediate_size": 512 * 2,
    "num_head_modal": 8,
    "num_layer_modal": 0
}

class My_model(nn.Module):
    def __init__(self,config):
        super(My_model,self).__init__()
        self.embedding = multimodal_embedding.MultimodalEmbedding(config)
        self.transformer = multimodal_transformer.Mul_Encoder(config)
        self.dense = nn.Linear(512, 512)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(512*3,512)
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512,1)
        self.jihuo = nn.Tanh()
        self.dropout = nn.Dropout(p=0.11)



    def forward(self , my_mask, visual_embedding , audio_embedding):


        output = self.embedding(visual_embedding , audio_embedding)

        output = self.transformer(output,my_mask)[0]
        #output = self.dense(output[:,0])
        # output = torch.cat((output[:,0], output[:,50],output[:,120]),dim=1)
        # output = self.activation(output)
        # output = self.dropout(output)
        # output = self.linear(output)
        # output = self.jihuo(output)
        #
        # output = self.linear2(output)
        # output = self.jihuo(output)*3
        return output


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

def build_my_mask_only_vis_imu(vis_len, imu_len):
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
    #return my_mask.reshape(1, -1)
    return torch.tensor(my_mask, dtype=torch.float32).unsqueeze(0)



if __name__ == '__main__':
    # 参数设定
    B = 4  # batch size
    vis_len = 120  # 视觉 token 数（不含 cls）
    imu_len = 10  # IMU token 数（不含 cls）
    hidden_size = 512  # 每个 token 的特征维度

    # 模拟视觉特征：比如来自 CNN 的 patch 特征
    visual_input = torch.randn(B, vis_len, hidden_size)  # 去掉1，是因为esp_1要加进来
    imu_input = torch.randn(B, imu_len, hidden_size)  # 同理

    my_mask = build_my_mask_only_vis_imu(vis_len, imu_len)

    # 实例化模块
    embedder = My_model(
        config
    )

    # 前向传播
    output = embedder(my_mask, visual_input, imu_input)
    print(visual_input.shape)
    print(imu_input.shape)
    print(my_mask.shape)
    # 输出 shape
    print("输出 shape:", output.shape)  # 应该是 [B, vis_len + imu_len, hidden_size]













