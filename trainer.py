import torch
import os
import networks
import logging
import json
import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import torch.optim as optim
from utils import sec_to_hm_str,readlines
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

config = {
    "imu_len": 11,
    "vis_len": 145,#121,257,145
    "hidden_size": 512,
    #"max_position": 150,
    "modal_size": 2,
    "layer_norm_eps": 1e-5,
    "hidden_dropout_prob": 0.1,         # dropout rate
    "visual_dim": 144,#120,256,144                   # 原始视觉特征维度
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
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class Trainer:
    def __init__(self, opts):
        logging.info('--------------------Mynew--------------------')
        self.opt = opts
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            logging.info('using adaptive depth binning!')
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
        else:
            logging.info('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)#-1
            if idx not in frames_to_load:
                frames_to_load.append(idx) #Loading frames: [0, -1, 1]

        logging.info('Loading frames: {}'.format(frames_to_load))

        # Model
        self.models["mutil_fram_encoder"] = networks.mutil_fram_en(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["mutil_fram_encoder"].to(self.device)

        self.models["mutil_fram_depth"] = networks.DepthDecoder(self.models["mutil_fram_encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["mutil_fram_depth"].to(self.device)

        # self.models["single_fram_encoder"] = networks.ResnetEncoder(18,
        #                                                             self.opt.weights_init == "pretrained")
        # self.models["single_fram_encoder"].to(self.device)
        #
        # self.models["single_fram_depth"] = networks.DepthDecoder(self.models["single_fram_encoder"].num_ch_enc, self.opt.scales)
        # self.models["single_fram_depth"].to(self.device)

        self.models["single_fram_depth"] = networks.SimmimMode
        self.models["single_fram_depth"].to(self.device)


        self.models["vis_pose_encoder"] = networks.ResnetEncoder(18,
                                                              self.opt.weights_init == "pretrained",
                                                              num_input_images=self.num_pose_frames)

        self.models["vis_pose_encoder"].to(self.device)

        self.models["imu_pose_encoder"] = networks.imupose_en()
        self.models["imu_pose_encoder"].to(self.device)

        self.models["vis_imu_encoder"] = networks.vis_imu_encoder()
        self.models["vis_imu_encoder"].to(self.device)

        self.models["vis_imu_fusemodel"] = networks.vis_imu_fusemodel(config)
        self.models["vis_imu_fusemodel"].to(self.device)

        self.models["pose_decoder_BiLSTM"] = networks.PoseDecoderBiLSTM()
        self.models["pose_decoder_BiLSTM"].to(self.device)


        self.parameters_to_train += list(self.models["mutil_fram_encoder"].parameters())
        self.parameters_to_train += list(self.models["mutil_fram_depth"].parameters())
        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["vis_pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["imu_pose_encoder"].parameters())
            #self.parameters_to_train += list(self.models["vis_imu_encoder"].parameters())
            self.parameters_to_train += list(self.models["vis_imu_fusemodel"].parameters())
            self.parameters_to_train += list(self.models["pose_decoder_BiLSTM"].parameters())
            #self.parameters_to_train += list(self.models["single_fram_encoder"].parameters())
            self.parameters_to_train += list(self.models["single_fram_depth"].parameters())



        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        logging.info('Training model named: {}'.format(self.opt.model_name))
        logging.info('Models and tensorboard events files are saved to: {}'.format(self.opt.log_dir))
        logging.info('Training is using device: {}'.format(self.device))

        # ------------------------------------weights-------------------------------------------------
        total_params = []
        total_params.append(sum(p.numel() for p in self.models["mutil_fram_encoder"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["mutil_fram_depth"].parameters()))
        if self.train_teacher_and_pose:
            total_params.append(sum(p.numel() for p in self.models["vis_pose_encoder"].parameters()))
            total_params.append(sum(p.numel() for p in self.models["imu_pose_encoder"].parameters()))
            #total_params.append(sum(p.numel() for p in self.models["vis_imu_encoder"].parameters()))
            total_params.append(sum(p.numel() for p in self.models["vis_imu_fusemodel"].parameters()))
            total_params.append(sum(p.numel() for p in self.models["pose_decoder_BiLSTM"].parameters()))
            total_params.append(sum(p.numel() for p in self.models["single_fram_depth"].parameters()))
            #total_params.append(sum(p.numel() for p in self.models["single_fram_encoder"].parameters()))



        total_params_weights = sum(total_params) / 1e6
        logging.info(f'--------------------weights: {total_params_weights}--------------------')
        # -----------------------------------优化--------------------------------------------------------
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)# NAdam AdamW
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        #=========================================================================
        # self.model_optimizer = optim.SGD(self.parameters_to_train,
        #                     lr=self.opt.learning_rate,  # 初始学习率
        #                     momentum=0.9,  # 动量项，加快收敛
        #                     weight_decay=1e-4,  # L2 正则化
        #                     nesterov=True  # Nesterov 动量（推荐）
        #                     )
        #
        # def poly_lr_lambda(epoch):
        #     return (1 - epoch / 20) ** 0.9
        #
        # self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(
        #     self.model_optimizer,
        #     lr_lambda=poly_lr_lambda
        # )
        # =========================================================================
        # self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)  # NAdam AdamW
        # self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #      self.model_optimizer, T_max=self.opt.num_epochs)#1e-6
        #-----------------------DATA-----------------------------------------------------------------

        # split_file = os.path.join(self.opt.data_path, "splits_3f.json")
        # train_filenames = readlines(split_file,"train")
        # val_filenames = readlines(split_file,"val")
        # test_filenames = readlines(split_file,"test")
        # logging.info(f"train 集加载成功，数量: {len(train_filenames)}")
        # logging.info(f"val 集加载成功，数量: {len(val_filenames)}")
        # logging.info(f"test 集加载成功，数量: {len(test_filenames)}")
        #
        #
        # train_datasets = datasets.VIDataset(self.opt.data_path, train_filenames, is_train=True)
        # val_datasets = datasets.VIDataset(self.opt.data_path, val_filenames,is_test = True)
        # test_datasets = datasets.VIDataset(self.opt.data_path, test_filenames,is_test = True)

        train_datasets = datasets.VIDataset(self.opt.data_path,Hsize=self.opt.height,Wsize=self.opt.width, split='train')
        val_datasets = datasets.VIDataset(self.opt.data_path,Hsize=self.opt.height,Wsize=self.opt.width, split='val',is_test=True)
        test_datasets = datasets.VIDataset(self.opt.data_path,Hsize=self.opt.height,Wsize=self.opt.width, split='test',is_test=True)

        self.train_loader = DataLoader(train_datasets, batch_size=self.opt.batch_size, shuffle=True,
                                       num_workers=self.opt.num_workers,
                                       pin_memory=True, drop_last=True,worker_init_fn=seed_worker)
        self.val_loader = DataLoader(val_datasets, batch_size=self.opt.batch_size, shuffle=False,
                                     num_workers=self.opt.num_workers,
                                     pin_memory=False, drop_last=True,worker_init_fn=seed_worker)
        self.test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False,
                                     num_workers= self.opt.num_workers ,
                                     pin_memory=False, drop_last=False,worker_init_fn=seed_worker)
        num_train_samples = len(train_datasets)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        logging.info(f'{self.num_total_steps}===============num_total_steps')
        self.val_iter = iter(self.val_loader)
        # -----------------------writer-----------------------------------------------------------------
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        # -----------------------other-----------------------------------------------------------------
        if not self.opt.no_ssim:
            self.ssim = networks.SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = networks.BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = networks.Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        #-----------------------------------------------------------------------------------------------
        self.save_opts()
    #===================================================================================================
    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            m.train()
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['mutil_fram_depth', 'mutil_fram_encoder']:
                    m.train()
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
    # ================================freeze===================================================================
    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            logging.info('freezing teacher and pose networks!')
            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["mutil_fram_encoder"].parameters())
            self.parameters_to_train += list(self.models["mutil_fram_depth"].parameters())
            self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()
    #=========================================================================================================
    def train(self):
        """Run the entire training pipeline
        """
        #torch.autograd.set_detect_anomaly(True)
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()
            self.run_epoch()
            self.test_epoch()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
    # =================================test================================================================
    def test_epoch(self):
        logging.info("============> Test {} <============".format(self.epoch))
        self.set_eval()

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        HEIGHT, WIDTH = self.opt.height, self.opt.width
        logging.info("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # 初始化帧索引列表
        frames_to_load = [0]
        if self.opt.use_future_frame:
            frames_to_load.append(1)
        frames_to_load.extend(
            [idx for idx in range(-1, -1 - self.opt.num_matching_frames, -1) if idx not in frames_to_load])

        # 初始化统计变量
        error_accumulator = np.zeros(7)  # 对应7个误差指标
        valid_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                # 1. 数据准备
                input_color = data[('color_aug', 0, 0)]
                input_depth = data['depth_gt'][:, 0]  # 保持tensor格式延迟转换

                # 2. 设备转移（使用non_blocking异步传输）
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                # 3. 模型推理
                if self.opt.eval_teacher:
                    # output = self.models["single_fram_encoder"](input_color)
                    # output = self.models["single_fram_depth"](output)
                    output = self.models["single_fram_depth"](input_color)
                else:
                    # 处理静态相机情况
                    if self.opt.static_camera:
                        for f_i in frames_to_load:
                            data[("color_aug", f_i, 0)] = data[('color_aug', 0, 0)]

                    # 姿态估计
                    pose_vis_raw = {f_i: data[("color_aug", f_i, 0)] for f_i in frames_to_load}
                    pose_imu_raw = data['imu']
                    my_mask = data['my_mask'][0]

                    if torch.cuda.is_available():
                        pose_vis_raw = {k: v.cuda() for k, v in pose_vis_raw.items()}
                        pose_imu_raw = pose_imu_raw.cuda()
                        my_mask = my_mask.cuda()

                    # 计算相对位姿
                    for fi in frames_to_load[1:]:
                        if fi < 0:
                            vis_feats = [
                                self.models["vis_pose_encoder"](torch.cat(
                                    [pose_vis_raw[fi], pose_vis_raw[fi + 1]], 1))
                            ]
                            imu_feats = self.models["imu_pose_encoder"](pose_imu_raw[:, :10, :])
                            fused_feat = self.models['vis_imu_fusemodel'](my_mask,vis_feats, imu_feats)
                            #vis_imu_fusemodel （my_mask,vis_feats, imu_feats）
                            # vis_imu_encoder

                            axisangle, translation = self.models['pose_decoder_BiLSTM'](fused_feat)
                            pose = networks.transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)

                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                            data[('relative_pose', fi)] = pose

                    # 多帧特征提取
                    lookup_frames = torch.stack([data[('color_aug', idx, 0)] for idx in frames_to_load[1:]], 1)
                    relative_poses = torch.stack([data[('relative_pose', idx)] for idx in frames_to_load[1:]], 1)

                    if torch.cuda.is_available():
                        lookup_frames = lookup_frames.cuda()
                        relative_poses = relative_poses.cuda()
                        K = data[('K', 2)].cuda()
                        invK = data[('inv_K', 2)].cuda()

                    features, _, _ = self.models["mutil_fram_encoder"](
                        input_color,
                        lookup_frames,
                        relative_poses,
                        K,
                        invK,
                        min_depth_bin=self.min_depth_tracker,
                        max_depth_bin=self.max_depth_tracker
                    )
                    output = self.models["mutil_fram_depth"](features)

                # 4. 计算视差和深度
                pred_disp, _ = networks.disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

                # 5. 转换为CPU numpy数组
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                batch_gt_depth = input_depth.numpy()

                # 6. 逐样本计算指标
                for i in range(pred_disp.shape[0]):
                    gt_depth = batch_gt_depth[i]
                    pred_depth = 1 / cv2.resize(pred_disp[i], (gt_depth.shape[1], gt_depth.shape[0]))

                    # 创建有效掩码
                    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                    #mask = gt_depth > 0
                    valid_gt = gt_depth[mask]
                    valid_pred = pred_depth[mask]

                    # 中值缩放（如果启用）
                    if not self.opt.disable_median_scaling:
                        ratio = np.median(valid_gt) / np.median(valid_pred)
                        valid_pred *= ratio

                    # 截断预测值
                    valid_pred = np.clip(valid_pred, MIN_DEPTH, MAX_DEPTH)
                    if len(valid_gt) == 0:
                        continue
                    # 计算并累加误差
                    errors = compute_errors(valid_gt, valid_pred)
                    error_accumulator += errors
                    valid_samples += 1

        # 8. 计算最终平均误差
        if valid_samples > 0:
            mean_errors = error_accumulator / valid_samples
            logging.info("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            logging.info(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        logging.info("\n-> Done!")
        self.set_train()
        return mean_errors if valid_samples > 0 else None

    # ======================================================================================================
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        logging.info("============> Training{} <============".format(self.epoch))

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            # 梯度裁剪：防止梯度爆炸
            #torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)

            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 500 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                self.log("train", inputs, outputs, losses)

                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            if self.step == self.opt.freeze_teacher_step:
                self.freeze_teacher()

            self.step += 1

        # 手动清理
        torch.cuda.empty_cache()

        self.model_lr_scheduler.step()
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, None)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None)

        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]  # b 4 4
        relative_poses = torch.stack(relative_poses, 1)  # b 1 4 4

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]  # 查找帧
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path
        if self.train_teacher_and_pose:
            # feats = self.models["single_fram_encoder"](inputs["color_aug", 0, 0])
            # mono_outputs.update(self.models['single_fram_depth'](feats))

            mono_outputs.update(self.models['single_fram_depth'](inputs["color_aug", 0, 0]))
        else:
            with torch.no_grad():
                # feats = self.models["single_fram_encoder"](inputs["color_aug", 0, 0])
                # mono_outputs.update(self.models['single_fram_depth'](feats))
                mono_outputs.update(self.models['single_fram_depth'](inputs["color_aug", 0, 0]))

        self.generate_images_pred(inputs, mono_outputs)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, lowest_cost, confidence_mask = self.models["mutil_fram_encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin)
        outputs.update(self.models["mutil_fram_depth"](features))

        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.opt.height, self.opt.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]

        if not self.opt.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                           self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val

        # update adaptive depth bins
        if self.train_teacher_and_pose:
            self.update_adaptive_depth_bins(outputs)

        return outputs, losses


    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_vis_raw = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            pose_imu_raw =  inputs['imu']
            for f_i in self.opt.frame_ids[1:]:

                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_vis_inputs = [pose_vis_raw[f_i], pose_vis_raw[0]]
                    pose_imu_inputs = pose_imu_raw[:, :10, :]
                else:
                    pose_vis_inputs = [pose_vis_raw[0], pose_vis_raw[f_i]]
                    pose_imu_inputs = pose_imu_raw[:, 10:, :]

                vis_pose_feats = [self.models["vis_pose_encoder"](torch.cat(pose_vis_inputs, 1))]
                imu_pose_feats = self.models["imu_pose_encoder"](pose_imu_inputs)
                fuse_pose_feat =self.models['vis_imu_fusemodel'](inputs['my_mask'][0],vis_pose_feats,imu_pose_feats)
                #vis_imu_fusemodel （inputs['my_mask'][0],vis_pose_feats,imu_pose_feats）
                # vis_imu_encoder

                axisangle, translation = self.models['pose_decoder_BiLSTM'](fuse_pose_feat)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = networks.transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_vis_raw = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            pose_imu_raw = inputs['imu']
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_vis_inputs = [pose_vis_raw[fi], pose_vis_raw[fi + 1]]
                        pose_imu_inputs = pose_imu_raw[:, :10, :]

                        vis_pose_feats = [self.models["vis_pose_encoder"](torch.cat(pose_vis_inputs, 1))]
                        imu_pose_feats = self.models["imu_pose_encoder"](pose_imu_inputs)
                        fuse_pose_feat = self.models['vis_imu_fusemodel'](inputs['my_mask'][0],vis_pose_feats, imu_pose_feats)
                        #vis_imu_fusemodel （inputs['my_mask'][0],vis_pose_feats, imu_pose_feats）
                        # vis_imu_encoder   （vis_pose_feats, imu_pose_feats）

                        axisangle, translation = self.models['pose_decoder_BiLSTM'](fuse_pose_feat)
                        pose = networks.transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_vis_inputs = [pose_vis_raw[fi - 1], pose_vis_raw[fi]]
                        pose_imu_inputs = pose_imu_raw[:, :10, :]

                        vis_pose_feats = [self.models["vis_pose_encoder"](torch.cat(pose_vis_inputs, 1))]
                        imu_pose_feats = self.models["imu_pose_encoder"](pose_imu_inputs)
                        fuse_pose_feat = self.models['vis_imu_fusemodel'](inputs['my_mask'][0],vis_pose_feats, imu_pose_feats)
                        #vis_imu_fusemodel (inputs['my_mask'][0],vis_pose_feats, imu_pose_feats)
                        # vis_imu_encoder （vis_pose_feats, imu_pose_feats）

                        axisangle, translation = self.models['pose_decoder_BiLSTM'](fuse_pose_feat)
                        pose = networks.transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_vis_raw[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            logging.info('----------- ONLY ACCEPT TWO FRAME INPUTS -----------')
            raise NotImplementedError

        return outputs
    def  generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = networks.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)


                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []


            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = networks.get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = torch.clamp(F.interpolate(
                 depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = networks.compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask
    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01
    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        logging.info(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                         sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in sorted(self.opt.frame_ids):
                writer.add_image(
                    "color_aug_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color_aug", frame_id, s)][j].data, self.step)#color,color_aug
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('mono_disp', s)][j, 0])
            writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

                consistency_mask = \
                    outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)
                writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                writer.add_image(
                    "lowest_cost_masked/{}".format(j),
                    lowest_cost * consistency_mask, self.step)
                writer.add_image(
                    "consistency_mask/{}".format(j),
                    consistency_mask, self.step)


                consistency_target = colormap(outputs["consistency_target/0"][j].squeeze(0))
                writer.add_image(
                    "consistency_target/{}".format(j),
                    consistency_target, self.step)
        # ----- 拼接三帧 color_aug 图像 -----
        # import torchvision.utils as vutils
        # triplet = []
        # for fid in [-1, 0, 1]:
        #     if (fid, s) in [(k[1], k[2]) for k in inputs.keys() if k[0] == "color_aug"]:
        #         triplet.append(inputs[("color_aug", fid, s)][j])
        #
        # if len(triplet) == 3:
        #     triplet_tensor = torch.cat(triplet, dim=2)  # 横向拼接在 W 维
        #     writer.add_image("triplet_color_aug/{}".format(j), triplet_tensor, self.step)
        #
        #     # 可选：保存本地图像（注释掉也可以）
        #     vutils.save_image(triplet_tensor, f"logs/triplet_color_aug_{mode}_{self.step}_b{j}.png")

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'mutil_fram_encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
    def load_model(self):
        """Load model(s) from disk
        """

        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'mutil_fram_encoder':
                if self.opt.fix_bins:
                    min_depth_bin = self.opt.min_depth
                    max_depth_bin = self.opt.max_depth
                else:
                    min_depth_bin = pretrained_dict.get('min_depth_bin')
                    max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    if not self.opt.fix_bins:
                        self.min_depth_tracker = min_depth_bin
                        self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            print(f"model({len(model_dict)}), weights({len(pretrained_dict)})")

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    def load_mono_model(self):

        model_list = ['vis_pose_encoder', 'imu_pose_encoder', 'vis_imu_fusemodel', 'pose_decoder_BiLSTM',
                       'single_fram_depth'] #vis_imu_fusemodel vis_imu_encoder 'single_fram_encoder',
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            print(f"model({len(model_dict)}), weights({len(pretrained_dict)})")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis