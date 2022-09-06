import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Cityscapes
from datasets import Config as DatasetConfig
from datasets import Kitti, Robotcar
from depth_pose_prediction.config import DepthPosePrediction as Config
from depth_pose_prediction.networks import (
    SSIM,
    BackprojectDepth,
    DepthDecoder,
    PoseDecoder,
    Project3D,
    ResnetEncoder,
)
from depth_pose_prediction.utils import (
    disp_to_depth,
    h_concat_images,
    transformation_from_parameters,
)

# matplotlib.use('Agg')


class DepthPosePrediction:
    def __init__(self, dataset_config: DatasetConfig, config: Config, use_online: bool = False):
        # Initialize parameters ===========================
        self.config_file = config.config_file
        self.dataset_type = dataset_config.dataset
        self.dataset_path = dataset_config.dataset_path
        self.height = dataset_config.height
        self.width = dataset_config.width
        self.train_set = config.train_set
        self.val_set = config.val_set
        self.resnet = config.resnet
        self.resnet_pretrained = config.resnet_pretrained
        self.scales = config.scales
        self.learning_rate = config.learning_rate
        self.scheduler_step_size = config.scheduler_step_size
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.num_epochs = config.num_epochs
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        self.disparity_smoothness = config.disparity_smoothness
        self.velocity_loss_scaling = config.velocity_loss_scaling
        self.mask_dynamic = config.mask_dynamic
        self.log_path = config.log_path
        self.save_frequency = config.save_frequency
        self.save_val_depth = config.save_val_depth
        self.save_val_depth_batches = config.save_val_depth_batches
        self.multiple_gpus = config.multiple_gpus
        self.gpu_ids = config.gpu_ids
        self.load_weights_folder = config.load_weights_folder
        self.use_wandb = False

        # Internal parameters =============================
        self.is_trained = False

        # Fixed parameters ================================
        self.frame_ids = (0, -1, 1)
        self.num_pose_frames = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Deal with dependent parameters ==================
        if self.load_weights_folder is not None:
            self.load_weights_folder = Path(self.load_weights_folder).absolute()

        if isinstance(self.train_set, list):
            self.train_set = tuple(self.train_set)
        if isinstance(self.val_set, list):
            self.val_set = tuple(self.val_set)
        if dataset_config.dataset == 'Kitti':
            if isinstance(self.val_set, int):
                self.val_set = (self.val_set, )
            if isinstance(self.train_set, str):
                if self.train_set != 'all':
                    raise ValueError('train_set of KITTI only accepts these strings: ["all"]')
                self.train_set = tuple([  # pylint: disable=consider-using-generator
                    s for s in range(11) if s not in self.val_set and s != 3  # No IMU for seq 3
                ])
            elif isinstance(self.train_set, int):
                self.train_set = (self.train_set, )
            if not (isinstance(self.train_set, tuple) and isinstance(self.train_set[0], int)):
                raise ValueError('Passed invalid value for train_set')
            if not (isinstance(self.val_set, tuple) and isinstance(self.val_set[0], int)):
                raise ValueError('Passed invalid value for val_set')
        elif dataset_config.dataset in ['Cityscapes', 'RobotCar']:
            if isinstance(self.train_set, str):
                self.train_set = (self.train_set, )
            if isinstance(self.val_set, str):
                self.val_set = (self.val_set, )
            if not (isinstance(self.train_set, tuple) and isinstance(self.train_set[0], str)):
                raise ValueError('Passed invalid value for train_set')
            if not (isinstance(self.val_set, tuple) and isinstance(self.val_set[0], str)):
                raise ValueError('Passed invalid value for val_set')

        if self.multiple_gpus and not torch.cuda.is_available():
            raise ValueError('Activated multiple GPUs but running on a CPU.')
        if self.multiple_gpus and self.gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        elif self.multiple_gpus:
            if any(i >= torch.cuda.device_count() for i in self.gpu_ids):
                raise ValueError('Passed invalid GPU ID.')
        if not self.multiple_gpus and self.gpu_ids is not None and len(self.gpu_ids) > 1:
            raise ValueError('Passed multiple GPU IDs without activating multi-GPU support.')
        if self.gpu_ids is not None and torch.cuda.is_available():
            # Set the main GPU for gradient averaging etc.
            torch.cuda.set_device(self.gpu_ids[0])
        elif self.gpu_ids is None and torch.cuda.is_available():
            self.gpu_ids = (0, )
        # =================================================

        # Construct networks ==============================
        self.models = {}
        self.models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
        self.models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                    self.scales)
        self.models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
                                                    self.num_pose_frames)
        self.models['pose_decoder'] = PoseDecoder(self.models['pose_encoder'].num_ch_encoder,
                                                  num_input_features=1,
                                                  num_frames_to_predict_for=2)

        self.use_online = use_online
        self.online_models = {}
        if self.use_online:
            self.online_models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
            self.online_models['depth_decoder'] = DepthDecoder(
                self.online_models['depth_encoder'].num_ch_encoder, self.scales)
            self.online_models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
                                                               self.num_pose_frames)
            self.online_models['pose_decoder'] = PoseDecoder(
                self.models['pose_encoder'].num_ch_encoder,
                num_input_features=1,
                num_frames_to_predict_for=2)
        # =================================================

        # Create the projected (warped) image =============
        self.backproject_depth = {}
        self.project_3d = {}
        self.backproject_depth_single = {}
        self.project_3d_single = {}
        for scale in self.scales:
            h = self.height // (2**scale)
            w = self.width // (2**scale)
            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.project_3d[scale] = Project3D(self.batch_size, h, w)

            self.backproject_depth_single[scale] = BackprojectDepth(1, h, w)
            self.project_3d_single[scale] = Project3D(1, h, w)
        # =================================================

        # Structural similarity ===========================
        self.ssim = SSIM()
        self.ssim.to(self.device)
        # =================================================

        # Send everything to the GPU(s) ===================
        if 'cuda' in self.device.type:
            print(f'Selected GPUs: {list(self.gpu_ids)}')
        if self.multiple_gpus:
            for model_name, m in self.models.items():
                if m is not None:
                    self.models[model_name] = nn.DataParallel(m, device_ids=self.gpu_ids)

        self.parameters_to_train = []
        for model_name, m in self.models.items():
            if m is not None:
                m.to(self.device)
                self.parameters_to_train += list(m.parameters())
        self.online_parameters_to_train = []
        for m in self.online_models.values():
            m.to(self.device)
            self.online_parameters_to_train += list(m.parameters())
        for m in self.backproject_depth.values():
            m.to(self.device)
        for m in self.project_3d.values():
            m.to(self.device)
        for m in self.backproject_depth_single.values():
            m.to(self.device)
        for m in self.project_3d_single.values():
            m.to(self.device)
        # =================================================

        # Set up optimizer ================================
        self.optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, 0.1)
        self.epoch = 0
        if use_online:
            self.online_optimizer = optim.Adam(self.online_parameters_to_train, self.learning_rate)
        else:
            self.online_optimizer = None
        # =================================================

        # Construct datasets ==============================
        self.train_loader, self.val_loader = None, None
        # =================================================

    # ============================================================
    # Training / validation

    def train(self,
              validate: bool = False,
              depth_error: bool = False,
              pose_error: bool = False,
              dataloader: Optional[DataLoader] = None,
              verbose: bool = True,
              use_wandb: Optional[bool] = False) -> None:
        use_wandb = False if use_wandb is None else use_wandb
        if use_wandb:
            self.use_wandb = True
            self._init_wandb()

        if dataloader is None:
            self._create_dataloaders(validation=validate)
        else:
            validate = False
            self.train_loader = dataloader
            print(f'Training samples:   {len(self.train_loader):>5}')

        # Training loop
        step = 0
        starting_epoch = self.epoch + 1
        for self.epoch in range(starting_epoch, self.num_epochs + 1):
            # Run a single epoch
            self._set_train()
            loss = []
            with tqdm(unit='batches',
                      total=len(self.train_loader),
                      desc=f'Training epoch {self.epoch}/{self.num_epochs}',
                      disable=not verbose) as pbar:
                for batch_i, sample_i in enumerate(self.train_loader):
                    # Run a single step
                    outputs, losses = self._process_batch(sample_i)
                    loss.append(losses['loss'].item())

                    self.optimizer.zero_grad()
                    losses['loss'].backward()
                    self.optimizer.step()

                    if self.use_wandb:
                        wandb.log(losses)

                    pbar.set_postfix(loss=np.mean(loss))
                    pbar.update(1)
                    step += 1
            self.lr_scheduler.step()
            self.is_trained = True

            if self.save_frequency > 0 and self.epoch % self.save_frequency == 0:
                self.save_model()

            if validate:
                validation_loss = self.validate()
                if self.use_wandb:
                    wandb.log({'validation_loss': validation_loss}, commit=False)

            if depth_error:
                error = self.compute_depth_error(median_scaling=True, print_results=False)
                if self.use_wandb:
                    wandb.log(error, commit=False)
            if pose_error:
                error = self.compute_pose_error(print_results=False)
                if self.use_wandb:
                    wandb.log(error, commit=False)

            if self.use_wandb:
                wandb.log({'training_loss': np.mean(loss), 'epoch': self.epoch})

        # Save the final model
        if self.save_frequency > -1:
            self.save_model()

    def adapt(self,
              inputs: Dict[Any, Tensor],
              online_index: int = 0,
              steps: int = 1,
              online_loss_weight: Optional[float] = None,
              use_expert: bool = True,
              do_adapt: bool = True):
        if online_loss_weight is None:
            loss_weights = None
        elif self.batch_size == 1:
            loss_weights = torch.ones(1, device=self.device)
        else:
            loss_weights = torch.empty(self.batch_size, device=self.device)
            buffer_loss_weight = (1 - online_loss_weight) / (self.batch_size - 1)
            loss_weights[online_index] = online_loss_weight
            loss_weights[np.arange(self.batch_size) != online_index] = buffer_loss_weight

        if do_adapt:
            self._set_adapt(freeze_encoder=True)
        else:
            self._set_eval()
            steps = 1

        for _ in range(steps):
            outputs, losses = self._process_batch(inputs, loss_weights)
            if do_adapt:
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()

        if self.batch_size != 1 and use_expert:
            online_inputs = {key: value[online_index].unsqueeze(0) for key, value in inputs.items()}
            for _ in range(steps):
                online_outputs, online_losses = self._process_batch(online_inputs, use_online=True)
                self.online_optimizer.zero_grad()
                online_losses['loss'].backward()
                self.online_optimizer.step()
            outputs = online_outputs
            losses = online_losses

        return outputs, losses

    def validate(self) -> float:
        """ Compute the validation loss(es)
        """
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)
        if self.val_loader is None:
            self._create_dataloaders(training=False)

        self._set_eval()
        loss = []
        with torch.no_grad(), tqdm(unit='batches', total=len(self.val_loader),
                                   desc='Validation') as pbar:
            for batch_i, sample_i in enumerate(self.val_loader):
                outputs, losses = self._process_batch(sample_i)
                loss.append(losses['loss'].item())

                if self.save_val_depth and batch_i < self.save_val_depth_batches:
                    self.save_prediction(sample_i, outputs)

                pbar.set_postfix(loss=np.mean(loss))
                pbar.update(1)
        return float(np.mean(loss))

    def compute_depth_error(
        self,
        median_scaling: bool = True,
        print_results: bool = True,
    ) -> Dict[str, float]:
        """ Compute error metrics for depth prediction
        Follows monodepth2 implementation:
        https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py

        monodepth2 on Kittti
        - cap depth at 80 per standard practice
        - per-image median ground truth scaling (or same for entire test set)
        - post-process: predict for original and flipped images, then combine both disparities
        - pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        """
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)

        if self.dataset_type == 'Kitti':
            dataset = Kitti(self.dataset_path,
                            self.val_set,
                            frame_ids=[0],
                            scales=[0],
                            height=self.height,
                            width=self.width,
                            with_depth=True)
        elif self.dataset_type == 'Cityscapes':
            dataset = Cityscapes(self.dataset_path,
                                 self.val_set,
                                 frame_ids=[0],
                                 scales=[0],
                                 height=self.height,
                                 width=self.width,
                                 with_depth=True)
        else:
            warnings.warn(f'Unsupported dataset: {self.dataset_type}', RuntimeWarning)
            return {}

        data_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
        print(f'Validation samples: {len(dataset):>5}')

        ratios = []
        num_samples = 0
        abs_diff, abs_rel, sq_rel, a1, a2, a3, rmse_tot, rmse_log_tot = 0, 0, 0, 0, 0, 0, 0, 0

        self._set_eval()
        with torch.no_grad(), tqdm(unit='batches', total=len(data_loader),
                                   desc='Validation') as pbar:
            for batch_i, sample_i in enumerate(data_loader):
                for key, val in sample_i.items():
                    sample_i[key] = val.to(self.device)
                gt_depth = sample_i[('depth', 0, -1)].squeeze().cpu().detach().numpy()
                gt_height, gt_width = gt_depth.shape

                # Depth prediction
                outputs = self._predict_disparity(sample_i)
                disparity = outputs[('disp', 0)].squeeze().cpu().detach().numpy()
                pred_depth = disp_to_depth(disparity, self.min_depth, None)
                pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

                # Mask out pixels without ground truth depth
                # or ground truth depth farther away than the maximum predicted depth
                if self.max_depth is not None:
                    mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
                else:
                    mask = gt_depth > self.min_depth
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                # Introduced by SfMLearner
                if median_scaling:
                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio

                # Cap predicted depth at min and max depth
                pred_depth[pred_depth < self.min_depth] = self.min_depth
                if self.max_depth is not None:
                    pred_depth[pred_depth > self.max_depth] = self.max_depth

                # Compute error metrics
                thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
                a1 += np.mean(thresh < 1.25)
                a2 += np.mean(thresh < 1.25**2)
                a3 += np.mean(thresh < 1.25**3)
                rmse = (gt_depth - pred_depth)**2
                rmse_tot += np.sqrt(np.mean(rmse))
                rmse_log = (np.log(gt_depth) - np.log(pred_depth))**2
                rmse_log_tot += np.sqrt(np.mean(rmse_log))
                abs_diff += np.mean(np.abs(gt_depth - pred_depth))
                abs_rel += np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
                sq_rel += np.mean(((gt_depth - pred_depth)**2) / gt_depth)

                num_samples += 1
                pbar.update(1)

        metrics = {
            'abs_diff': abs_diff / num_samples,
            'abs_rel': abs_rel / num_samples,
            'sq_rel': sq_rel / num_samples,
            'a1': a1 / num_samples,
            'a2': a2 / num_samples,
            'a3': a3 / num_samples,
            'rmse': rmse_tot / num_samples,
            'rmse_log': rmse_log_tot / num_samples
        }

        if print_results:
            for key, value in metrics.items():
                print(f'{key:<8}: {value:>6.3f}')

        if median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            metrics['med_scaling'] = med
            if print_results:
                print(f'Scaling ratios | med: {med:.3f} | std: {np.std(ratios / med):.3f}')

        return metrics

    def compute_pose_error(self, print_results: bool = True) -> Dict[str, float]:
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)

        if self.dataset_type == 'Kitti':
            dataset = Kitti(self.dataset_path,
                            self.val_set,
                            frame_ids=[-1, 0],
                            scales=[0],
                            height=self.height,
                            width=self.width,
                            poses=True,
                            with_depth=True)
        else:
            warnings.warn(f'Unsupported dataset: {self.dataset_type}', RuntimeWarning)
            return {}
        data_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
        print(f'Validation samples: {len(dataset):>5}')

        num_samples = 0
        rpe_trans, rpe_rot = 0, 0

        self._set_eval()
        with torch.no_grad(), tqdm(unit='batches', total=len(data_loader),
                                   desc='Validation') as pbar:
            for batch_i, sample_i in enumerate(data_loader):
                image_0 = sample_i['rgb_aug', -1, 0]
                image_1 = sample_i['rgb_aug', 0, 0]
                pred_transformation, _ = self.predict_pose(image_0, image_1, as_numpy=False)
                pred_transformation = pred_transformation.squeeze().detach().cpu()
                pred_transformation = torch.linalg.inv(pred_transformation)

                gt_transformation = sample_i['relative_pose', 0].squeeze()
                rel_err = torch.linalg.inv(gt_transformation) @ pred_transformation
                trans_error = torch.linalg.norm(rel_err[:3, 3]).item()
                rot_error = np.arccos(
                    max(min(.5 * (torch.trace(rel_err[:3, :3]).item() - 1), 1.), -1.0))

                rpe_trans += trans_error
                rpe_rot += rot_error * 180 / np.pi

                num_samples += 1
                pbar.update(1)

        metrics = {'rpe_trans': rpe_trans / num_samples, 'rpe_rot': rpe_rot / num_samples}

        if print_results:
            for key, value in metrics.items():
                print(f'{key:<8}: {value:>6.3f}')

        return metrics

    # ============================================================
    # Predict functions

    def predict(self, batch) -> Dict[Any, Tensor]:
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)
        self._set_eval()
        with torch.no_grad():
            outputs, losses = self._process_batch(batch)
        return outputs

    def predict_from_image(self, image, as_numpy: bool = True):
        """ Take one image as input and return the predicted depth
        """
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)
        self._set_eval()
        with torch.no_grad():
            image = image.to(self.device)

            # Depth network
            features = self.models['depth_encoder'](image)
            disp = self.models['depth_decoder'](features)[('disp', 0)]
            depth = disp_to_depth(disp, self.min_depth, self.max_depth)

        if as_numpy:
            depth = depth.squeeze().cpu().detach().numpy()
        return depth

    def predict_from_images(
        self,
        image_0: Tensor,
        image_1: Tensor,
        as_numpy: bool = True,
        return_loss: bool = False,
        camera_matrix: Optional[Tensor] = None,
        inv_camera_matrix: Optional[Tensor] = None,
        relative_distance: Optional[Tensor] = None,
    ):
        """ Take two images as input and return depth for both and relative pose
        """
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)

        if len(image_0.shape) == 3:
            image_0 = image_0.unsqueeze(dim=0)
        if len(image_1.shape) == 3:
            image_1 = image_1.unsqueeze(dim=0)

        self._set_eval()
        with torch.no_grad():
            image_0 = image_0.to(self.device)
            image_1 = image_1.to(self.device)

            # Depth network
            features_0 = self.models['depth_encoder'](image_0)
            disp_0 = self.models['depth_decoder'](features_0)
            depth_0 = disp_to_depth(disp_0[('disp', 0)], self.min_depth, self.max_depth)
            features_1 = self.models['depth_encoder'](image_1)
            disp_1 = self.models['depth_decoder'](features_1)
            depth_1 = disp_to_depth(disp_1[('disp', 0)], self.min_depth, self.max_depth)

            # Pose network
            pose_inputs = torch.cat([image_0, image_1], 1)
            pose_features = [self.models['pose_encoder'](pose_inputs)]
            axis_angle, translation = self.models['pose_decoder'](pose_features)
            transformation = transformation_from_parameters(axis_angle[:, 0],
                                                            translation[:, 0],
                                                            invert=False)

        if as_numpy:
            depth_0 = depth_0.squeeze().cpu().detach().numpy()
            depth_1 = depth_1.squeeze().cpu().detach().numpy()
            transformation = transformation.squeeze().cpu().detach().numpy()

        if return_loss:
            # Assume: image_0 => frame -1 | image_1 => frame 0
            frame_ids = self.frame_ids
            self.frame_ids = (0, -1)
            outputs = disp_1
            outputs[('axis_angle', 0, -1)] = axis_angle[:, 0]
            outputs[('translation', 0, -1)] = translation[:, 0]
            outputs[('cam_T_cam', 0, -1)] = transformation_from_parameters(axis_angle[:, 0],
                                                                           translation[:, 0],
                                                                           invert=True)
            inputs = {
                ('rgb', -1, 0): image_0,
                ('rgb', 0, 0): image_1,
                ('camera_matrix', 0): camera_matrix.to(self.device),
                ('inv_camera_matrix', 0): inv_camera_matrix.to(self.device),
                ('relative_distance', 0): relative_distance.to(self.device)
            }
            self._reconstruct_images(inputs, outputs)
            losses = self._compute_loss(inputs, outputs, scales=(0, ))
            for k, v in losses.items():
                losses[k] = v.squeeze().cpu().detach().numpy()
            self.frame_ids = frame_ids
            return depth_0, depth_1, transformation, losses

        return depth_0, depth_1, transformation

    def predict_pose(
        self,
        image_0: Tensor,
        image_1: Tensor,
        as_numpy: bool = True,
        use_online: bool = False,
    ) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
        if not self.is_trained:
            warnings.warn('The model has not been trained yet.', RuntimeWarning)

        if len(image_0.shape) == 3:
            image_0 = image_0.unsqueeze(dim=0)
        if len(image_1.shape) == 3:
            image_1 = image_1.unsqueeze(dim=0)

        self._set_eval()
        with torch.no_grad():
            image_0 = image_0.to(self.device)
            image_1 = image_1.to(self.device)

            # Pose network
            pose_inputs = torch.cat([image_0, image_1], 1)
            if use_online:
                pose_features = [self.online_models['pose_encoder'](pose_inputs)]
                axis_angle, translation = self.online_models['pose_decoder'](pose_features)
            else:
                pose_features = [self.models['pose_encoder'](pose_inputs)]
                axis_angle, translation = self.models['pose_decoder'](pose_features)
            axis_angle, translation = axis_angle[:, 0], translation[:, 0]
            transformation = transformation_from_parameters(axis_angle, translation, invert=False)

            cov_matrix = torch.eye(6, device=self.device)

        if as_numpy:
            transformation = transformation.squeeze().cpu().detach().numpy()
            cov_matrix = cov_matrix.cpu().detach().numpy()
        return transformation, cov_matrix

    # ============================================================
    # Save / load functions

    def save_model(self) -> None:
        """Save model weights to disk
        """
        save_folder = self.log_path / 'models' / f'weights_{self.epoch:03}'
        save_folder.mkdir(parents=True, exist_ok=True)

        # Save the network weights
        for model_name, model in self.models.items():
            if model is None:
                continue
            save_path = save_folder / f'{model_name}.pth'
            if isinstance(model, nn.DataParallel):
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            if 'encoder' in model_name:
                # ToDo: look into this
                # Save the sizes - these are needed at prediction time
                to_save['height'] = Tensor(self.height)
                to_save['width'] = Tensor(self.width)
            torch.save(to_save, save_path)

        # Save the optimizer and the LR scheduler
        optimizer_save_path = save_folder / 'optimizer.pth'
        to_save = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(to_save, optimizer_save_path)

        # Save the config file
        config_save_path = self.log_path / 'config.yaml'
        shutil.copy(self.config_file, config_save_path)

        print(f'Saved model to: {save_folder}')

    def load_model(self, load_optimizer: bool = True) -> None:
        """Load model(s) from disk
        """
        if self.load_weights_folder is None:
            print('Weights folder required to load the model is not specified.')
        if not self.load_weights_folder.exists():
            print(f'Cannot find folder: {self.load_weights_folder}')
        print(f'Load model from: {self.load_weights_folder}')

        # Load the network weights
        for model_name, model in self.models.items():
            if model is None:
                continue
            path = self.load_weights_folder / f'{model_name}.pth'
            pretrained_dict = torch.load(path, map_location=self.device)
            if isinstance(model, nn.DataParallel):
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict.keys()) == 0:
                raise RuntimeError(f'No fitting weights found in: {path}')
            model_dict.update(pretrained_dict)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_dict)
            else:
                model.load_state_dict(model_dict)
        self.is_trained = True

        if load_optimizer:
            # Load the optimizer and LR scheduler
            optimizer_load_path = self.load_weights_folder / 'optimizer.pth'
            try:
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                if 'optimizer' in optimizer_dict:
                    self.optimizer.load_state_dict(optimizer_dict['optimizer'])
                    self.lr_scheduler.load_state_dict(optimizer_dict['scheduler'])
                    self.epoch = self.lr_scheduler.last_epoch
                    print(f'Restored optimizer and LR scheduler (resume from epoch {self.epoch}).')
                else:
                    self.optimizer.load_state_dict(optimizer_dict)
                    print('Restored optimizer (legacy mode).')
            except:  # pylint: disable=bare-except
                print('Cannot find matching optimizer weights, so the optimizer is randomly '
                      'initialized.')

    def load_online_model(self, load_optimizer: bool = True) -> None:
        """Load model(s) from disk
        """
        if self.load_weights_folder is None:
            print('Weights folder required to load the model is not specified.')
        if not self.load_weights_folder.exists():
            print(f'Cannot find folder: {self.load_weights_folder}')
        print(f'Load online model from: {self.load_weights_folder}')

        # Load the network weights
        for model_name, model in self.online_models.items():
            if model is None:
                continue
            path = self.load_weights_folder / f'{model_name}.pth'
            pretrained_dict = torch.load(path, map_location=self.device)
            if isinstance(model, nn.DataParallel):
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict.keys()) == 0:
                raise RuntimeError(f'No fitting weights found in: {path}')
            model_dict.update(pretrained_dict)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_dict)
            else:
                model.load_state_dict(model_dict)

        if load_optimizer:
            # Load the optimizer and LR scheduler
            optimizer_load_path = self.load_weights_folder / 'optimizer.pth'
            try:
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                if 'optimizer' in optimizer_dict:
                    self.online_optimizer.load_state_dict(optimizer_dict['optimizer'])
                    print('Restored online optimizer')
                else:
                    self.online_optimizer.load_state_dict(optimizer_dict)
                    print('Restored online optimizer (legacy mode).')
            except:  # pylint: disable=bare-except
                print('Cannot find matching optimizer weights, so the optimizer is randomly '
                      'initialized.')

    # ============================================================
    # Auxiliary functions

    def _set_train(self) -> None:
        for m in self.models.values():
            if m is not None:
                m.train()

    def _set_eval(self) -> None:
        for m in self.models.values():
            if m is not None:
                m.eval()

    def _set_adapt(self, freeze_encoder: bool = True) -> None:
        """ Set all to train except for batch normalization (freeze parameters)
        Convert all models to adaptation mode: batch norm is in eval mode + frozen params
        Adapted from:
        https://github.com/Yevkuzn/CoMoDA/blob/main/code/CoMoDA.py
        """
        for model_name, model in self.models.items():
            model.eval()  # To set the batch norm to eval mode
            for name, param in model.named_parameters():
                if name.find('bn') != -1:
                    param.requires_grad = False  # Freeze batch norm
                if freeze_encoder and 'encoder' in model_name:
                    param.requires_grad = False

        for model_name, model in self.online_models.items():
            model.eval()  # To set the batch norm to eval mode
            for name, param in model.named_parameters():
                if name.find('bn') != -1:
                    param.requires_grad = False  # Freeze batch norm
                if freeze_encoder and 'encoder' in model_name:
                    param.requires_grad = False

    def _create_dataloaders(self, training: bool = True, validation: bool = True):
        valid_dataset_types = ['Kitti', 'Cityscapes', 'RobotCar']
        if self.dataset_type not in valid_dataset_types:
            raise ValueError(f'dataset_type must be one of {valid_dataset_types}')
        if self.train_set is not None and training:
            if self.dataset_type == 'Kitti':
                train_dataset = Kitti(self.dataset_path,
                                      self.train_set,
                                      self.frame_ids,
                                      self.scales,
                                      self.height,
                                      self.width,
                                      do_augmentation=True,
                                      views=('left', 'right'),
                                      with_mask=self.mask_dynamic)
            elif self.dataset_type == 'Cityscapes':
                train_dataset = Cityscapes(self.dataset_path,
                                           self.train_set,
                                           self.frame_ids,
                                           self.scales,
                                           self.height,
                                           self.width,
                                           do_augmentation=True,
                                           with_mask=self.mask_dynamic)
            else:  # self.dataset_type == 'RobotCar':
                train_dataset = Robotcar(self.dataset_path,
                                         self.train_set,
                                         self.frame_ids,
                                         self.scales,
                                         self.height,
                                         self.width,
                                         do_augmentation=True,
                                         with_mask=self.mask_dynamic,
                                         start_frame=4000,
                                         end_frame=24000)
            print(f'Training samples:   {len(train_dataset):>5}')
            self.train_loader = DataLoader(train_dataset,
                                           self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           drop_last=True)
        if self.val_set is not None and validation:
            if self.dataset_type == 'Kitti':
                val_dataset = Kitti(self.dataset_path,
                                    self.val_set,
                                    self.frame_ids,
                                    self.scales,
                                    self.height,
                                    self.width,
                                    with_mask=self.mask_dynamic)
            elif self.dataset_type == 'Cityscapes':
                val_dataset = Cityscapes(self.dataset_path,
                                         self.val_set,
                                         self.frame_ids,
                                         self.scales,
                                         self.height,
                                         self.width,
                                         with_mask=self.mask_dynamic)
            else:  # self.dataset_type == 'RobotCar':
                val_dataset = Robotcar(self.dataset_path,
                                       self.val_set,
                                       self.frame_ids,
                                       self.scales,
                                       self.height,
                                       self.width,
                                       with_mask=self.mask_dynamic,
                                       start_frame=500,
                                       end_frame=4000)
            print(f'Validation samples: {len(val_dataset):>5}')
            self.val_loader = DataLoader(val_dataset,
                                         self.batch_size,
                                         shuffle=False,
                                         num_workers=self.num_workers,
                                         pin_memory=True,
                                         drop_last=True)

    def _process_batch(
        self,
        inputs: Dict[Any, Tensor],
        loss_sample_weights: Optional[Tensor] = None,
        use_online: bool = False,
    ) -> Tuple[Dict[Any, Tensor], Dict[str, Tensor]]:
        """
        Pass a minibatch through the network
        """

        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        outputs = {}
        outputs.update(self._predict_disparity(inputs, use_online=use_online))
        outputs.update(self._predict_poses(inputs, use_online=use_online))
        self._reconstruct_images(inputs, outputs)  # also converts disparity to depth
        losses = self._compute_loss(inputs, outputs, sample_weights=loss_sample_weights)
        return outputs, losses

    def _predict_disparity(self,
                           inputs: Dict[Any, Tensor],
                           frame: int = 0,
                           scale: int = 0,
                           use_online: bool = False) -> Dict[Any, Tensor]:
        if use_online:
            features = self.online_models['depth_encoder'](inputs[('rgb_aug', frame, scale)])
            outputs = self.online_models['depth_decoder'](features)
        else:
            features = self.models['depth_encoder'](inputs[('rgb_aug', frame, scale)])
            outputs = self.models['depth_decoder'](features)
        return outputs

    def _predict_poses(self,
                       inputs: Dict[Any, Tensor],
                       use_online: bool = False) -> Dict[Any, Tensor]:
        """
        Predict the poses: 0 -> -1 and 0 -> 1
        """
        assert self.num_pose_frames == 2, self.num_pose_frames
        assert self.frame_ids == (0, -1, 1)

        outputs = {}
        pose_inputs_dict = {f_i: inputs['rgb_aug', f_i, 0] for f_i in self.frame_ids}
        for frame_i in self.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if frame_i < 0:
                pose_inputs = [pose_inputs_dict[frame_i], pose_inputs_dict[0]]
            else:
                pose_inputs = [pose_inputs_dict[0], pose_inputs_dict[frame_i]]
            pose_inputs = torch.cat(pose_inputs, 1)
            if use_online:
                pose_features = [self.online_models['pose_encoder'](pose_inputs)]
                axis_angle, translation = self.online_models['pose_decoder'](pose_features)
            else:
                if self.models['pose_encoder'] is None:
                    axis_angle, translation = self.models['pose_decoder'](pose_inputs)
                else:
                    pose_features = [self.models['pose_encoder'](pose_inputs)]
                    axis_angle, translation = self.models['pose_decoder'](pose_features)
            axis_angle, translation = axis_angle[:, 0], translation[:, 0]
            outputs[('axis_angle', 0, frame_i)] = axis_angle
            outputs[('translation', 0, frame_i)] = translation

            # Invert the matrix such that it is always frame 0 -> frame X
            outputs[('cam_T_cam', 0,
                     frame_i)] = transformation_from_parameters(axis_angle,
                                                                translation,
                                                                invert=(frame_i < 0))
        return outputs

    def _reconstruct_images(
        self,
        inputs: Dict[Any, Tensor],
        outputs: Dict[Any, Tensor],
    ) -> None:
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are added to the 'outputs' dictionary.
        """
        batch_size = outputs['disp', self.scales[0]].shape[0]

        for scale in self.scales:
            # Upsample the disparity from scale to the target height x width
            disp = outputs[('disp', scale)]
            disp = F.interpolate(disp, [self.height, self.width],
                                 mode='bilinear',
                                 align_corners=False)
            source_scale = 0

            depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[('depth', scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs[('cam_T_cam', 0, frame_id)]

                if batch_size == 1:
                    cam_points = self.backproject_depth_single[source_scale](
                        depth, inputs[('inv_camera_matrix', source_scale)])
                    pixel_coordinates = self.project_3d_single[source_scale](
                        cam_points, inputs[('camera_matrix', source_scale)], T)
                else:
                    cam_points = self.backproject_depth[source_scale](depth,
                                                                      inputs[('inv_camera_matrix',
                                                                              source_scale)])
                    pixel_coordinates = self.project_3d[source_scale](cam_points,
                                                                      inputs[('camera_matrix',
                                                                              source_scale)], T)
                # Save the warped image
                outputs[('rgb', frame_id, scale)] = F.grid_sample(inputs[('rgb', frame_id,
                                                                          source_scale)],
                                                                  pixel_coordinates,
                                                                  padding_mode='border',
                                                                  align_corners=True)

    def _compute_loss(
        self,
        inputs: Dict[Any, Tensor],
        outputs: Dict[Any, Tensor],
        scales: Optional[Tuple[int, ...]] = None,
        sample_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the losses for a minibatch.
        """
        assert self.frame_ids in ((0, -1, 1), (0, -1))
        scales = self.scales if scales is None else scales

        if sample_weights is None:
            sample_weights = torch.ones(self.batch_size, device=self.device) / self.batch_size

        source_scale = 0
        losses = {}
        total_loss = torch.zeros(1, device=self.device)

        for scale in scales:
            # Compute reprojection loss for every scale ========
            target = inputs['rgb', 0, source_scale]
            reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs['rgb', frame_id, scale]
                reprojection_losses.append(self._compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            # ==================================================

            # Auto-masking =====================================
            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs['rgb', frame_id, source_scale]
                identity_reprojection_losses.append(self._compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            # Add random numbers to break ties
            identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape,
                                                        device=self.device) * 0.00001
            combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)

            # "minimum among computed losses allows for robust reprojection"
            # https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf
            to_optimize, _ = torch.min(combined, dim=1)

            # Mask potentially dynamic objects =================
            if self.mask_dynamic:
                # 0: dynamic; 1: static
                mask = 1 - inputs['mask', 0, source_scale].squeeze()
                mask = mask.type(torch.bool)
                to_optimize = torch.masked_select(to_optimize, mask)  # Also flattens the array
            # ==================================================

            # Total self-supervision (reprojection) loss =======
            if not self.mask_dynamic:
                reprojection_loss = (to_optimize.mean(2).mean(1) * sample_weights).sum()
            else:
                reprojection_loss = to_optimize.mean()  # pre-training with masks
            losses[f'reprojection_loss/scale_{scale}'] = reprojection_loss
            # ==================================================

            # Compute smoothness loss for every scale ==========
            if self.mask_dynamic:
                mask = 1 - inputs['mask', 0, scale]
                mask = mask.type(torch.bool)
            else:
                mask = torch.ones_like(outputs['disp', scale], dtype=torch.bool, device=self.device)
            color = inputs['rgb', 0, scale]
            disp = outputs['disp', scale]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = self._compute_smooth_loss(norm_disp, color, mask)
            smooth_loss = (smooth_loss * sample_weights).sum()
            losses[f'smooth_loss/scale_{scale}'] = smooth_loss
            # ==================================================

            regularization_loss = self.disparity_smoothness / (2**scale) * smooth_loss
            losses[f'reg_loss/scale_{scale}'] = regularization_loss
            # ==================================================

            loss = reprojection_loss + regularization_loss
            losses[f'depth_loss/scale_{scale}'] = loss
            total_loss += loss
        total_loss /= len(self.scales)
        losses['depth_loss'] = total_loss

        # Velocity supervision loss (scale independent) ====
        if self.velocity_loss_scaling is not None and self.velocity_loss_scaling > 0:
            velocity_loss = self.velocity_loss_scaling * self._compute_velocity_loss(
                inputs, outputs)
            velocity_loss = (velocity_loss * sample_weights).sum()
            losses['velocity_loss'] = velocity_loss
            total_loss += velocity_loss
        # ==================================================

        losses['loss'] = total_loss

        if np.isnan(losses['loss'].item()):
            for k, v in losses.items():
                print(k, v.item())
            raise RuntimeError('NaN loss')

        return losses

    # ============================================================
    # Losses

    def _compute_velocity_loss(
        self,
        inputs: Dict[Any, Tensor],
        outputs: Dict[Any, Tensor],
    ) -> Tensor:
        batch_size = inputs['index'].shape[0]  # might be different from self.batch_size
        velocity_loss = torch.zeros(batch_size, device=self.device).squeeze()
        num_frames = 0
        for frame in self.frame_ids:
            if frame == -1:
                continue
            if frame == 0:
                pred_translation = outputs[('translation', 0, -1)]
            else:  # frame == 1
                pred_translation = outputs[('translation', 0, 1)]
            gt_distance = torch.abs(inputs[('relative_distance', frame)]).squeeze()
            pred_distance = torch.linalg.norm(pred_translation, dim=-1).squeeze()
            velocity_loss += F.l1_loss(pred_distance, gt_distance,
                                       reduction='none')  # separated by sample in batch
            num_frames += 1
        velocity_loss /= num_frames
        return velocity_loss

    @staticmethod
    def _compute_smooth_loss(
        disp: Tensor,
        img: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        grad_disp_x = torch.masked_select(grad_disp_x, mask[..., :-1])
        grad_disp_y = torch.masked_select(grad_disp_y, mask[..., :-1, :])

        batch_size = disp.shape[0]
        smooth_loss = torch.empty(batch_size, device=disp.device)
        for i in range(batch_size):
            _grad_disp_x = torch.masked_select(grad_disp_x[i, ...], mask[i, :, :, :-1])
            _grad_disp_y = torch.masked_select(grad_disp_y[i, ...], mask[i, :, :-1, :])
            smooth_loss[i] = _grad_disp_x.mean() + _grad_disp_y.mean()

        return smooth_loss

    def _compute_reprojection_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Computes reprojection loss between a batch of predicted and target images
        This is the photometric error
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    # ============================================================
    # Logging

    def save_prediction(
        self,
        inputs: Dict[Any, Tensor],
        outputs: Dict[Any, Tensor],
    ) -> None:
        save_folder = self.log_path / 'prediction' / f'val_depth_{self.epoch:03}'
        save_folder.mkdir(parents=True, exist_ok=True)

        pil_image = None

        # Iterate over the images
        for i, index in enumerate(inputs['index']):
            rgb = inputs['rgb', 0, 0][i]
            depth = outputs['depth', 0][i]

            rgb_np = rgb.squeeze().cpu().detach().numpy()
            rgb_np = np.moveaxis(rgb_np, 0, -1)  # Convert from [c, h, w] to [h, w, c]
            depth_np = depth.squeeze().cpu().detach().numpy()

            fig = plt.figure(figsize=(12.8, 9.6))
            plt.subplot(211)
            plt.imshow(rgb_np)
            plt.title('Input')
            plt.axis('off')
            plt.subplot(212)
            vmax = np.percentile(depth_np, 95)
            plt.imshow(depth_np, cmap='magma_r', vmax=vmax)
            plt.title(f'Depth prediction  |  vmax={vmax:.3f}')
            plt.axis('off')

            save_file = save_folder / f'{index.item():05}.png'
            # fig.suptitle(str(save_file)[-50:])
            fig.canvas.draw()

            if pil_image is None:
                pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                            fig.canvas.tostring_rgb())
            elif pil_image.size[0] < 5 * self.width:
                pil_image = h_concat_images(
                    pil_image,
                    Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb()))

            plt.savefig(save_file, bbox_inches='tight')
            plt.close()

        if self.use_wandb:
            wandb.log({'pred_depth': [wandb.Image(pil_image)]})

    def _init_wandb(self):
        # Name of the run as shown in the wandb GUI
        name = self.log_path.name.replace('log_', '')

        wandb.init(project='CL-SLAM', name=name)
        wandb.config.dataset_type = self.dataset_type
        wandb.config.train_set = self.train_set
        wandb.config.val_set = self.val_set
        wandb.config.height = self.height
        wandb.config.width = self.width
        wandb.config.batch_size = self.batch_size
        wandb.config.num_workers = self.num_workers
        wandb.config.resnet = self.resnet
        wandb.config.learning_rate = self.learning_rate
        wandb.config.scheduler_step_size = self.scheduler_step_size
        wandb.config.min_depth = self.min_depth
        wandb.config.max_depth = self.max_depth
        wandb.config.disparity_smoothness = self.disparity_smoothness
        wandb.config.velocity_loss_scaling = self.velocity_loss_scaling
        wandb.config.mask_dynamic = self.mask_dynamic
        wandb.config.log_path = self.log_path
