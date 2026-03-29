from .vis_pose_encoder import ResnetEncoder
from .imu_pose_encoder import IMUEncoder as imupose_en
from  .visualinertialmodel import ModalityFusionTransformer as vis_imu_encoder
from .posedecoderBiLSTM import PoseDecoderBiLSTM
from .vis_imu_model import My_T_MAAM as vis_imu_fusemodel
from .mutil_fram_resnet_encoder import ResnetEncoderMatching as mutil_fram_en
from .depth_decoder import DepthDecoder
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors


# Define a Swin Transformer for SimMIM
from .transformer_depth import build_simmim

simmim_config = {
    'model_type': 'swin',
    'img_size': 384,#512,#,#(480,640) 384
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 0,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 4,#8,#4,#2,#5
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'ape': False,
    'patch_norm': True,
    'use_checkpoint': False,
}
SimmimMode = build_simmim(simmim_config,pretrained=True)