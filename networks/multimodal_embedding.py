import torch
from torch import nn
# from transformers import BertModel, BertConfig, BertTokenizer,BertPreTrainedModel
# import numpy as np
# from transformer4onemodal import Mul_Encoder



class MultimodalEmbedding(nn.Module):
    """
    构建word_embedding, position_embedding , modal_embedding
    """
    def __init__(self, config):
        super(MultimodalEmbedding , self).__init__()
        self.modal_size = config['modal_size']         #模态数
        self.vis_len = config['vis_len']
        self.imu_len = config['imu_len']

        #self.position_embedding = nn.Embedding(int(config['max_position']/config['modal_size']) , config['hidden_size'])
        self.posi_visual_embedding = nn.Embedding(self.vis_len, config['hidden_size'])
        self.posi_imu_embedding = nn.Embedding(self.imu_len, config['hidden_size'])
        self.modal_embedding = nn.Embedding(self.modal_size , config['hidden_size'])

        #self.bert = BertModel.from_pretrained('bert-base-uncased' , output_hidden_states = True )

        self.esp_1 = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))  #20 35
        self.esp_2 = nn.Parameter(torch.zeros(1, 1 , config['hidden_size'])) #5  74
        #self.fc_visual = nn.Linear(config['visual_dim'] , config['word_embedding_dim'])
        #self.dropout_v = nn.Dropout(p=0.1)
        #self.fc_audio = nn.Linear(config['audio_dim'], config['word_embedding_dim'])
        #self.dropout_a = nn.Dropout(p=0.1)

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])


        #self.encoder_v = Mul_Encoder(config)
        #self.encoder_a = Mul_Encoder(config)
        ###################################


    def forward(self , visual_embedding , audio_embedding):
        size_of_batch = visual_embedding.size(0)

        #cls_token = self.cls_token.repeat(size_of_batch, 1, 1)
        esp_1 = self.esp_1.repeat(size_of_batch, 1, 1)
        esp_2 = self.esp_2.repeat(size_of_batch, 1, 1)

        visual_embedding = torch.cat([esp_1, visual_embedding], dim=1)
        imu_embedding = torch.cat([esp_2, audio_embedding], dim=1)


        #word_embedding = torch.cat([cls_token , word_embedding] , dim= 1)
        # word_embedding = self.bert(input_ids = input_ids , attention_mask = input_mask,token_type_ids = segment_ids).hidden_states[12]
        # visual_embedding = self.fc_visual(visual_embedding)
        # audio_embedding = self.fc_audio(audio_embedding)

        #visual position embedding
        visual_position_id = torch.arange(self.vis_len, dtype= torch.long , device = visual_embedding.device).unsqueeze(0)
        visual_position_id = visual_position_id.repeat(size_of_batch, 1)
        visual_position_embedding = self.posi_visual_embedding(visual_position_id)

        #imu position embedding
        imu_position_id = torch.arange(self.imu_len, dtype= torch.long , device = imu_embedding.device).unsqueeze(0)
        imu_position_id = imu_position_id.repeat(size_of_batch, 1)
        imu_position_embedding = self.posi_imu_embedding(imu_position_id)

        visual_embedding = visual_embedding + visual_position_embedding
        imu_embedding = imu_embedding + imu_position_embedding

        #visual_embedding = self.encoder_v(visual_embedding,input_mask)
        #audio_embedding = self.encoder_a(audio_embedding,input_mask)


        input_embedding = torch.cat([visual_embedding, imu_embedding] , dim=1)
        # modal_embedding
        modal_ids_0 = torch.zeros(size=(size_of_batch, self.vis_len), dtype= torch.long ,device = input_embedding.device)
        modal_ids_1 = torch.ones(size= (size_of_batch, self.imu_len), dtype= torch.long ,device = input_embedding.device)

        modal_ids = torch.cat([modal_ids_0, modal_ids_1] , dim=1)
        modal_embedding = self.modal_embedding(modal_ids)

        embedding = modal_embedding + input_embedding
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)

        return embedding



# config = {
#     "imu_len": 10,
#     "vis_len": 120,
#     "hidden_size": 512,
#     #"max_position": 150,
#     "modal_size": 2,
#     "layer_norm_eps": 1e-5,
#     "hidden_dropout_prob": 0.1,         # dropout rate
#     "visual_dim": 120,                   # 原始视觉特征维度
#     "imu_dim": 10,                     # 原始音频特征维度
#     "attention_dropout_prob": 0.1,
#     "num_head": 4,
#     "output_attention": 0,
#     "num_layer": 3,
#     "output_hidden_state": 0,
#     "intermediate_size": 512 * 2,
#     "num_head_modal": 8,
#     "num_layer_modal": 0
# }
# # 参数设定
# B = 4                 # batch size
# vis_len = 120         # 视觉 token 数（不含 cls）
# imu_len = 10          # IMU token 数（不含 cls）
# hidden_size = 512     # 每个 token 的特征维度
#
# # 模拟视觉特征：比如来自 CNN 的 patch 特征
# visual_input = torch.randn(B, vis_len, hidden_size)  # 去掉1，是因为esp_1要加进来
# imu_input = torch.randn(B, imu_len, hidden_size)      # 同理
#
# # 实例化模块
# embedder = MultimodalEmbedding(
# config
# )
#
# # 前向传播
# output = embedder(visual_input, imu_input)
#
# # 输出 shape
# print("输出 shape:", output.shape)  # 应该是 [B, vis_len + imu_len, hidden_size]







