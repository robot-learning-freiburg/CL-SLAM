import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Kitti, Robotcar
from depth_pose_prediction import DepthPosePrediction
from loop_closure_detection import LoopClosureDetection
from slam.pose_graph_optimization import PoseGraphOptimization
from slam.replay_buffer import ReplayBuffer
from slam.utils import calc_depth_error, rotation_error, translation_error

PLOTTING = True


class Slam:
    def __init__(self, config):
        # Config =========================================
        self.config = config
        self.online_dataset_type = config.dataset.dataset
        self.online_dataset_path = config.dataset.dataset_path
        self.do_adaptation = config.slam.adaptation
        self.adaptation_epochs = config.slam.adaptation_epochs
        self.min_distance = config.slam.min_distance
        self.start_frame = config.slam.start_frame
        self.logging = config.slam.logging
        if not self.do_adaptation:
            config.depth_pose.batch_size = 1
        self.log_path = config.depth_pose.log_path
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.do_loop_closures = config.slam.do_loop_closures
        self.keyframe_frequency = config.slam.keyframe_frequency
        self.lc_distance_poses = config.slam.lc_distance_poses

        # Depth / pose predictor ==========================
        self.predictor = DepthPosePrediction(config.dataset, config.depth_pose, use_online=False)
        self.predictor.load_model(load_optimizer=False)

        # Dataloader ======================================
        if self.online_dataset_type == 'Kitti':
            self.online_dataset = Kitti(
                self.online_dataset_path,
                config.slam.dataset_sequence,
                config.dataset.frame_ids,
                config.dataset.scales,
                config.dataset.height,
                config.dataset.width,
                poses=True,  # Ground truth poses
                with_depth=False,
                min_distance=config.slam.min_distance,
            )
        elif self.online_dataset_type == 'RobotCar':
            if config.slam.dataset_sequence == 1:
                start_frame, end_frame = 750, 4750
            else:
                start_frame, end_frame = 22100, 26100
            self.online_dataset = Robotcar(
                self.online_dataset_path,
                '2015-08-12-15-04-18',
                config.dataset.frame_ids,
                config.dataset.scales,
                config.dataset.height,
                config.dataset.width,
                poses=True,  # Ground truth poses
                min_distance=config.slam.min_distance,
                start_frame=start_frame,
                end_frame=end_frame,
                every_n_frame=2,
            )
        else:
            raise ValueError('Unsupported dataset type.')
        self.online_dataloader = DataLoader(
            self.online_dataset,
            batch_size=1,  # Simulates online loading
            shuffle=False,  # Simulates online loading
            num_workers=config.depth_pose.num_workers,
            pin_memory=True,
            drop_last=True)
        self.online_dataloader_iter = iter(self.online_dataloader)

        if self.do_adaptation and config.depth_pose.batch_size > 1: # and False:
            replay_buffer_path = config.replay_buffer.load_path
            replay_buffer_path.mkdir(parents=True, exist_ok=True)
            replay_buffer_state_path = replay_buffer_path / 'buffer_state.pkl'
            replay_buffer_state_path = replay_buffer_state_path if \
                replay_buffer_state_path.exists() else None
            self.replay_buffer = ReplayBuffer(
                replay_buffer_path,
                self.online_dataset_type,
                replay_buffer_state_path,
                self.online_dataset.height,
                self.online_dataset.width,
                self.online_dataset.scales,
                self.online_dataset.frame_ids,
                do_augmentation=True,
                batch_size=config.depth_pose.batch_size - 1,
                maximize_diversity=config.replay_buffer.maximize_diversity,
                max_buffer_size=config.replay_buffer.max_buffer_size,
                similarity_threshold=config.replay_buffer.similarity_threshold,
                similarity_sampling=config.replay_buffer.similarity_sampling,
            )
        else:
            self.replay_buffer = None

        # Pose graph backend ==============================
        self.loop_closure_detection = LoopClosureDetection(config.loop_closure)
        self.pose_graph = PoseGraphOptimization()
        if self.start_frame == 0:
            self.pose_graph.add_vertex(0, self.online_dataset.global_poses[1], fixed=True)
            # self.pose_graph.add_vertex(0, np.eye(4), fixed=True)
        self.gt_pose_graph = PoseGraphOptimization()  # Used for visualization
        self.gt_pose_graph.add_vertex(0, self.online_dataset.global_poses[1], fixed=True)

        # Auxiliary variables =============================
        self.current_step = 0
        self.since_last_loop_closures = self.lc_distance_poses

        # Logging =========================================
        # Track the relative error per step
        self.rel_trans_error = []
        self.rel_rot_error = []
        # Track the losses of the online data
        self.depth_loss = []
        self.velocity_loss = []
        # Track the depth error per step
        self.depth_error = []

        self.depth_loss_threshold = -1  # .04
        self.velo_loss_threshold = -1

    def __len__(self):
        return len(self.online_dataset)

    def step(self):
        self.current_step += 1

        # Combine online and replay data ==================
        online_data = next(self.online_dataloader_iter)

        self.predictor._set_eval()
        with torch.no_grad():
            online_image = online_data['rgb', 0, 0].to(self.predictor.device)
            online_features = self.predictor.models['depth_encoder'](online_image)[4].detach()
            online_features = online_features.mean(-1).mean(-1).cpu().numpy()

        if self.replay_buffer is not None:
            self.replay_buffer.add(online_data,
                                   self.online_dataset.get_item_filenames(self.current_step - 1),
                                   online_features,
                                   verbose=True)
        if self.replay_buffer is not None:
            replay_data = self.replay_buffer.get(online_data, online_features)
            if replay_data:
                training_data = self._cat_dict(online_data, replay_data)
            else:
                training_data = online_data
        else:
            training_data = online_data
        # =================================================

        # Use the measured velocity for this check
        if self.current_step > 1 and online_data['relative_distance',
                                                 1] < self.min_distance:
            print(f'skip: {online_data["relative_distance", 1].detach().cpu().numpy()[0]}')
            return {'depth_loss': 0, 'velocity_loss': 0}

        # Depth / pose prediction =========================
        # The returned losses are wrt the online data
        if self.do_adaptation:
            # Update the network weights
            outputs, losses = self.predictor.adapt(online_data,
                                                   training_data,
                                                   steps=self.adaptation_epochs)
        else:
            outputs, losses = self.predictor.adapt(online_data, None)
        # Extract input/output for online data
        image = online_data['rgb', 1, 0]
        if torch.sign(online_data['relative_distance', 1]) < 0:
            transformation = outputs['cam_T_cam', 0, 1][0, :]
        else:
            transformation = torch.linalg.inv(outputs['cam_T_cam', 0, 1][0, :])
        # Move to CPU for further processing
        transformation = transformation.squeeze().cpu().detach().numpy()
        for k, v in losses.items():
            losses[k] = float(v.squeeze().cpu().detach().numpy())
        if 'velocity_loss' not in losses:
            losses['velocity_loss'] = 0
        if 'depth_loss' not in losses:
            losses['depth_loss'] = 0
        # =================================================

        # Ground truth poses ==============================
        gt_transformation = online_data['relative_pose', 1].squeeze().cpu().detach().numpy()
        gt_pose = online_data['absolute_pose', 1].squeeze().cpu().detach().numpy()
        self.gt_pose_graph.add_vertex(self.current_step, gt_pose)
        self.gt_pose_graph.add_edge((self.gt_pose_graph.vertex_ids[-2], self.current_step),
                                    gt_transformation)
        # =================================================

        # Pose graph ======================================
        # Mapping can start later to account for initial warming up to the dataset
        if self.current_step == self.start_frame:
            self.pose_graph.add_vertex(self.current_step, gt_pose, fixed=True)
            print(f'Start mapping at frame {self.current_step}')
        elif self.current_step > self.start_frame:
            # Initialize with predicted odometry
            odom_pose = self.pose_graph.get_pose(self.pose_graph.vertex_ids[-1]) @ transformation
            self.pose_graph.add_vertex(self.current_step, odom_pose)
            cov_matrix = np.eye(6)
            cov_matrix[2, 2] = .1
            cov_matrix[5, 5] = .1
            self.pose_graph.add_edge((self.pose_graph.vertex_ids[-2], self.current_step),
                                     transformation,
                                     information=np.linalg.inv(cov_matrix))
        # =================================================

        # Loop closure detection ==========================
        optimized = False
        if self.do_loop_closures and self.current_step >= self.start_frame:
            self.loop_closure_detection.add(self.current_step, image.squeeze())
            if not self.current_step % self.keyframe_frequency and self.current_step < 4000:
                if self.since_last_loop_closures > self.lc_distance_poses:
                    lc_step_ids, distances = self.loop_closure_detection.search(self.current_step)
                    for i, d in zip(lc_step_ids, distances):
                        lc_image = self.online_dataset[i - 1]['rgb', 1, 0]
                        lc_transformation, cov_matrix = self.predictor.predict_pose(image,
                                                                                    lc_image,
                                                                                    as_numpy=True)
                        graph_transformation = self.pose_graph.get_transform(self.current_step, i)
                        print(f'{self.current_step} --> {i} '
                              f'[sim={d:.3f}, pred_dist={np.linalg.norm(lc_transformation):.1f}m, '
                              f'graph_dist={np.linalg.norm(graph_transformation):.1f}m]')
                        # LoopClosureDetection.display_matches(image, lc_image, self.current_step,
                        #                                      i, lc_transformation, d)
                        cov_matrix = np.eye(6)
                        cov_matrix[2, 2] = .1
                        cov_matrix[5, 5] = .1
                        self.pose_graph.add_edge((self.current_step, i),
                                                 lc_transformation,
                                                 information=.5 * np.linalg.inv(cov_matrix),
                                                 is_loop_closure=True)
                    if len(lc_step_ids) > 0:
                        self.pose_graph.optimize(max_iterations=10000, verbose=False)
                        optimized = True
            if optimized:
                self.since_last_loop_closures = 0
            else:
                self.since_last_loop_closures += 1
        # =================================================

        # Track metrics ===================================
        if self.logging:
            # Relative error of prediction
            rel_err = np.linalg.inv(gt_transformation) @ transformation
            self.rel_trans_error.append(translation_error(rel_err))
            self.rel_rot_error.append(rotation_error(rel_err))
            # Loss
            self.depth_loss.append(losses['depth_loss'])
            self.velocity_loss.append(losses['velocity_loss'])
            # Depth error
            if self.online_dataset_type == 'Kitti':
                self.depth_error.append(
                    calc_depth_error(outputs['depth', 0][0, ...].squeeze().detach().cpu().numpy(),
                                     online_data['depth', 0,
                                                 -1][0, ...].squeeze().detach().cpu().numpy(),
                                     min_depth=self.predictor.min_depth,
                                     max_depth=self.predictor.max_depth))
            # Plot the tracked metrics
            if PLOTTING and (not self.current_step % 100 or optimized):
                self.plot_metrics()
                self.plot_trajectory()
                self.pose_graph.visualize_in_meshlab(self.log_path / 'pose_graph.obj',
                                                     verbose=False)
                self.gt_pose_graph.visualize_in_meshlab(self.log_path / 'gt_pose_graph.obj',
                                                        verbose=False)
        # =================================================

        return losses

    def save_metrics(self) -> None:
        data = {
            'rel_trans_error': self.rel_trans_error,
            'rel_rot_error': self.rel_rot_error,
            'depth_loss': self.depth_loss,
            'velocity_loss': self.velocity_loss,
            'depth_error': self.depth_error,
        }
        filename = self.log_path / 'metrics.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def save_model(self) -> None:
        self.predictor.save_model()
        if self.replay_buffer is not None:
            self.replay_buffer.save_state()

    @staticmethod
    def _cat_dict(dict_1, dict_2):
        """ Concatenate the elements of online input dictionary
        with the corresponding elements of the replay buffer dictionary
        """
        res_dict = {}
        for key in dict_1:
            if key in dict_2:
                res_dict[key] = torch.cat([dict_1[key], dict_2[key]])
        return res_dict

    @staticmethod
    def _pose_graph_to_2d_trajectory(pose_graph):
        # Returns the trajectory in X-Z dimension
        poses = pose_graph.get_all_poses()
        trajectory = np.asarray([[p[0, 3], p[2, 3]] for p in poses])
        return trajectory

    def plot_trajectory(self):
        pred_trajectory = self._pose_graph_to_2d_trajectory(self.pose_graph)
        gt_trajectory = self._pose_graph_to_2d_trajectory(self.gt_pose_graph)
        fig = plt.figure()
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], '--.', label='pred')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], '--.', label='gt')
        plt.axis('equal')
        plt.legend()
        plt.title(f'Step = {self.current_step}')
        # filename = self.log_path / 'trajectory' / f'step_{self.current_step:04}.png'
        # filename.parent.mkdir(parents=True, exist_ok=True)
        # plt.savefig(str(filename))
        filename = self.log_path / 'trajectory.png'
        plt.savefig(str(filename))
        plt.close(fig)
        np.save(self.log_path / 'trajectory.npy', pred_trajectory)
        np.save(self.log_path / 'gt_trajectory.npy', gt_trajectory)

    def plot_metrics(self, filename: str = 'metrics.png'):
        if self.depth_error:
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
        else:
            # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))
            fig, axs = plt.subplots(nrows=2, ncols=2)
        # Losses
        axs[0, 0].plot(self.depth_loss)
        axs[0, 0].axhline(self.depth_loss_threshold, color='r')
        axs[0, 0].set_ylim(bottom=0, top=1.1 * max(self.depth_loss))
        axs[0, 0].set_xlabel('Step')
        axs[0, 0].set_ylabel('Depth loss')
        axs[0, 0].set_title('Depth loss')
        axs[1, 0].plot(self.velocity_loss)
        axs[1, 0].axhline(self.velo_loss_threshold, color='r')
        axs[1, 0].set_ylim(bottom=0, top=1.1 * max(self.velocity_loss))
        axs[1, 0].set_xlabel('Step')
        axs[1, 0].set_ylabel('Velocity loss')
        axs[1, 0].set_title('Velocity loss')
        # Relative errors
        axs[0, 1].plot(self.rel_trans_error)
        axs[0, 1].set_ylim(bottom=0)
        axs[0, 1].set_xlabel('Step')
        axs[0, 1].set_ylabel('Relative trans. error')
        axs[0, 1].set_title('Relative trans. error')
        axs[1, 1].plot(self.rel_rot_error)
        axs[1, 1].set_ylim(bottom=0)
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Relative rot. error')
        axs[1, 1].set_title('Relative rot. error')
        # Depth error
        if self.depth_error:
            axs[0, 2].plot([x['abs_rel'] for x in self.depth_error])
            axs[0, 2].set_ylim(bottom=0)
            axs[0, 2].set_xlabel('Step')
            axs[0, 2].set_ylabel('Abs rel')
            axs[0, 2].set_title('Abs rel / ARD')
            axs[1, 2].plot([x['sq_rel'] for x in self.depth_error])
            axs[1, 2].set_ylim(bottom=0)
            axs[1, 2].set_xlabel('Step')
            axs[1, 2].set_ylabel('Sq rel')
            axs[1, 2].set_title('Sq rel / SRD')
            axs[0, 3].plot([x['rmse'] for x in self.depth_error])
            axs[0, 3].set_ylim(bottom=0)
            axs[0, 3].set_xlabel('Step')
            axs[0, 3].set_ylabel('RMSE')
            axs[0, 3].set_title('RMSE')
            axs[1, 3].plot([x['a1'] for x in self.depth_error])
            axs[1, 3].set_ylim(top=1)
            axs[1, 3].set_xlabel('Step')
            axs[1, 3].set_ylabel('A1')
            axs[1, 3].set_title('A1')

        fig.tight_layout()
        plt.savefig(self.log_path / filename, bbox_inches='tight')
        plt.close(fig)
