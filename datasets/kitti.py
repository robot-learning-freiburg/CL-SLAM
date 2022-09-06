# Adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/datasets/kitti_dataset.py
# https://github.com/nianticlabs/monodepth2/blob/master/datasets/mono_dataset.py#L28

import argparse
from datetime import datetime
from os import PathLike
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from datasets.utils import Dataset


class Kitti(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path, PathLike],
        sequences: Union[int, List[int], Tuple[int, ...]],
        frame_ids: Union[List[int], Tuple[int, ...]],
        scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        do_augmentation: bool = False,
        views: Union[List[str], Tuple[str, ...]] = ('left', ),
        with_depth: bool = False,
        with_mask: bool = False,
        poses: bool = False,
        min_distance: float = 0,
    ) -> None:
        super().__init__(data_path,
                         frame_ids,
                         scales,
                         height,
                         width,
                         do_augmentation,
                         views,
                         with_depth,
                         with_mask,
                         min_distance=min_distance)

        # if self.with_depth and (sequences != 8 and sequences[0] != 8):
        #     raise ValueError('gt_depth only supported for sequence 8')

        self.include_poses = poses

        sequences = (sequences, ) if isinstance(sequences, int) else sequences
        if any(s > 10 for s in sequences):
            raise ValueError('Passed a sequence without ground truth data.')
        if 3 in sequences:
            raise ValueError('Passed a sequence without IMU data.')
        self.sequences = sorted(sequences)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        self._load_image_filenames()
        self.left_depth_filenames = self._get_filenames(
            mode='depth_left') if self.with_depth else []
        if self.with_mask:
            self._load_mask_filenames()
        self.velocity_filenames = self._get_filenames('velocity')
        self.timestamps = self._load_timestamps()
        self.global_poses = self._load_global_poses()

        # Make sure that the RGB images and the depth data matches
        # Adjust other data as well
        if self.with_depth:
            assert len(sequences) == 1  # ToDo: update sequence_indices
            depth_numbers = [int(d.stem) for d in self.left_depth_filenames]
            tmp_images = []
            tmp_velocity = []
            tmp_timestamps = []
            tmp_masks = []
            tmp_poses = []
            for i, img in enumerate(self.left_img_filenames):
                if int(img.stem) in depth_numbers:
                    tmp_images.append(img)
                    tmp_velocity.append(self.velocity_filenames[i])
                    tmp_timestamps.append(self.timestamps[i])
                    tmp_poses.append(self.global_poses[i])
                    if self.with_mask:
                        tmp_masks.append(self.left_mask_filenames[i])
            self.left_img_filenames = tmp_images
            self.velocity_filenames = tmp_velocity
            self.timestamps = tmp_timestamps
            self.global_poses = np.stack(tmp_poses)
            self.left_mask_filenames = tmp_masks
            self.sequence_indices[sequences[0]] = (0, len(self.left_img_filenames) - 1)

        # Filter data to meet minimum distance requirements
        self.relative_distances = self._compute_relative_distance()
        if self.min_distance > 0:
            assert len(sequences) == 1  # ToDo: update sequence_indices
            self._filter_by_distance(self.min_distance)

        self.relative_poses = self._load_relative_poses()  # Has to be done in the end

    def _get_filenames(self, mode: str) -> List[Path]:
        """Load the file names of the corresponding subfolders of a sequence.
        """
        valid_modes = [
            'rgb_left', 'rgb_right', 'lidar', 'depth_left', 'mask_left', 'mask_right', 'velocity'
        ]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}.')
        mode_dir = {
            'rgb_left': 'image_2',
            'rgb_right': 'image_3',
            'lidar': 'velodyne',
            'depth_left': 'gt_depth/image_02',
            'mask_left': 'segm_mask/image_2',
            'mask_right': 'segm_mask/image_3',
            'velocity': 'oxts/data',
        }[mode]
        mode_ext = {
            'rgb_left': 'png',
            'rgb_right': 'png',
            'lidar': 'bin',
            'depth_left': 'png',
            'mask_left': 'png',
            'mask_right': 'png',
            'velocity': 'txt',
        }[mode]
        filenames = []
        for sequence in self.sequences:
            sequence_index = [len(filenames)]
            sequence_data_path = self.data_path / 'sequences' / f'{sequence:02}' / mode_dir
            filenames += sorted(sequence_data_path.glob(f'*.{mode_ext}'))
            sequence_index.append(len(filenames) - 1)
            if sequence not in self.sequence_indices:
                self.sequence_indices[sequence] = tuple(sequence_index)
        return filenames

    def _load_timestamps(self) -> List[int]:
        timestamp_format = '%Y-%m-%d %H:%M:%S.%f'

        timestamps_ = []
        for sequence in self.sequences:
            # Load timestamps as written in the file, e.g., 2011-10-03 12:55:34.997992704
            timestamps_file = self.data_path / 'sequences' / f'{sequence:02}' / 'oxts' / \
                              'timestamps.txt'
            with open(timestamps_file, 'r', encoding='utf-8') as f:
                str_timestamps = f.read().splitlines()
            # Convert relative to timestamp of initial frame and output in seconds
            # Discard nanoseconds as they are not supported by datetime
            sequence_timestamps = []
            for timestamp in str_timestamps:
                sequence_timestamps.append(
                    (datetime.strptime(timestamp[:-3], timestamp_format) -
                     datetime.strptime(str_timestamps[0][:-3], timestamp_format)).total_seconds())
            timestamps_ += sequence_timestamps
        return timestamps_

    def _load_global_poses(self) -> np.ndarray:
        global_poses = []
        for sequence in self.sequences:
            # Load global poses
            poses_data_path = self.data_path / 'poses' / f'{sequence:02}.txt'
            sequence_poses = np.loadtxt(str(poses_data_path), dtype=np.float32).reshape((-1, 3, 4))
            # Convert to homogenous coordinates
            sequence_poses = np.concatenate(
                (sequence_poses, np.zeros((sequence_poses.shape[0], 1, 4), dtype=np.float32)), 1)
            sequence_poses[:, 3, 3] = 1
            global_poses.append(sequence_poses)
        return np.concatenate(global_poses)

    def _load_relative_poses(self) -> np.ndarray:
        relative_poses = []
        for sequence in self.sequences:
            indices = self.sequence_indices[sequence]
            sequence_poses = self.global_poses[indices[0]:indices[1] + 1]

            # Convert global poses to poses that are relative to the previous frame
            # The initial value is zero as there is no previous frame
            sequence_poses_relative = [np.eye(4, dtype=np.float32)]
            for i in range(1, len(sequence_poses)):
                sequence_poses_relative.append(
                    np.linalg.inv(sequence_poses[i - 1]) @ sequence_poses[i])
            sequence_poses_relative = np.stack(sequence_poses_relative)
            relative_poses.append(sequence_poses_relative)
        return np.concatenate(relative_poses)

    def _compute_relative_distance(self) -> List[float]:
        relative_distances = [0.]
        for index in range(1, len(self.timestamps)):
            relative_distances.append(self._load_relative_distance(index))
        return relative_distances

    def _filter_by_index(self, keep_indices: Union[np.ndarray, List[int]]) -> None:
        # Remove all timestamps, images, and velocities without poses
        self.left_img_filenames = [self.left_img_filenames[i]
                                   for i in keep_indices] if self.left_img_filenames else []
        self.right_img_filenames = [self.right_img_filenames[i]
                                    for i in keep_indices] if self.right_img_filenames else []
        self.left_mask_filenames = [self.left_mask_filenames[i]
                                    for i in keep_indices] if self.left_mask_filenames else []
        self.right_mask_filenames = [self.right_mask_filenames[i]
                                     for i in keep_indices] if self.right_mask_filenames else []
        self.left_depth_filenames = [self.left_depth_filenames[i]
                                     for i in keep_indices] if self.left_depth_filenames else []
        self.velocity_filenames = [self.velocity_filenames[i] for i in keep_indices]
        self.timestamps = [self.timestamps[i] for i in keep_indices]
        self.global_poses = [self.global_poses[i] for i in keep_indices]

    def _filter_by_distance(self, min_distance: float) -> None:
        distance = 0
        keep_indices = [0]
        relative_distances = [0]
        for i, relative_distance in enumerate(self.relative_distances[1:], start=1):
            distance += np.abs(relative_distance)
            if distance >= min_distance:
                keep_indices.append(i)
                relative_distances.append(distance)
                distance = 0
        self._filter_by_index(keep_indices)
        # Do no re-compute with function to exploit higher frequency
        self.relative_distances = relative_distances

    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        """
        ('rgb', <frame_id>, <scale>)
        ('camera_matrix', <scale>)
        ('inv_camera_matrix', <scale>)
        ('relative_pose', <frame_id>)        | with respect to frame_id - 1

        <frame_id> is relative to the requested index, i.e., the temporal neighbors
        <scale> == -1 denotes the original size and is deleted before returning
        """

        img_filenames, mask_filenames, index, do_color_augmentation, do_flip = self._pre_getitem(
            index)

        # Load the color images
        item = {('index'): index}
        original_image_shape = None
        for frame_id in self.frame_ids:
            rgb = Image.open(img_filenames[index + frame_id]).convert('RGB')
            if frame_id == 0:
                original_image_shape = rgb.size
            if do_flip:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            if len(self.scales) > 0:
                # Immediately rescale since the images are of various size across the sequences
                # Keeping the original resolution means that pytorch cannot batch images from
                #   different sequences
                rgb = self.resize[0](rgb)
                item[('rgb', frame_id, 0)] = rgb
            else:
                item[('rgb', frame_id, -1)] = rgb

        # Adjusting intrinsics to match each scale in the pyramid
        for scale in range(len(self.scales)):
            camera_matrix, inv_camera_matrix = self._scale_camera_matrix(self.camera_matrix, scale)
            item[('camera_matrix', scale)] = camera_matrix
            item[('inv_camera_matrix', scale)] = inv_camera_matrix
        if len(self.scales) == 0:
            # Load camera matrix of raw image data
            camera_matrix = self.camera_matrix.copy()
            camera_matrix[0, :] *= original_image_shape[0]
            camera_matrix[1, :] *= original_image_shape[1]
            item[('camera_matrix', -1)] = camera_matrix
            item[('inv_camera_matrix', -1)] = np.linalg.pinv(camera_matrix)

        # Load mask of potentially dynamic objects
        if self.with_mask:
            frame_id = 0
            mask = Image.open(mask_filenames[index + frame_id])
            if do_flip:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if len(self.scales) > 0:
                mask = self.resize[0](mask)
                item[('mask', frame_id, 0)] = mask
            else:
                item[('mask', frame_id, -1)] = mask

        # Load unscaled depth
        if self.with_depth:
            # Debugging checks
            for frame_id in self.frame_ids:
                assert int(img_filenames[index + frame_id].stem) == int(
                    self.left_depth_filenames[index + frame_id].stem)

            for frame_id in self.frame_ids:
                assert self.views == ('left', )
                depth = Image.open(self.left_depth_filenames[index + frame_id])
                if do_flip:
                    depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                item[('depth', frame_id, -1)] = depth

        # Load relative distance (except for first frame)
        for frame_id in self.frame_ids[1:]:
            item['relative_distance', frame_id] = self.relative_distances[index + frame_id]

        if self.include_poses:
            # Load the poses (except for the first frame)
            for frame_id in self.frame_ids[1:]:
                item[('relative_pose', frame_id)] = self.relative_poses[index + frame_id]
                if do_flip:
                    # We also have to flip the relative pose (rotate around y-axis)
                    item[('relative_pose', 0)][2, 0] *= -1
                    item[('relative_pose', 0)][0, 2] *= -1
                item[('absolute_pose', frame_id)] = self.global_poses[index + frame_id]

        self._post_getitem(item, do_color_augmentation)  # This will also call _preprocess()
        return item

    def _load_relative_distance(self, index: int) -> float:
        """
        Distance in meters and with respect to the previous frame.
        """
        previous_timestamp = self.timestamps[index - 1]
        current_timestamp = self.timestamps[index]
        delta_timestamp = current_timestamp - previous_timestamp  # s
        # Compute speed as norm of forward, leftward, and upward elements
        previous_speed = np.linalg.norm(np.loadtxt(str(self.velocity_filenames[index - 1]))[8:11])
        current_speed = np.linalg.norm(np.loadtxt(str(self.velocity_filenames[index]))[8:11])
        speed = (previous_speed + current_speed) / 2  # m/s
        distance = speed * delta_timestamp
        return distance

    def _preprocess(
        self,
        item: Dict[Any, Tensor],
        augment: Union[Callable, Tuple[Tensor, Optional[float], Optional[float], Optional[float],
                                       Optional[float]]],
    ) -> None:
        # Apply augmentation
        # Convert to list object as we are changing the size during the iteration
        for key in list(item.keys()):
            value = item[key]
            if 'rgb' in key:
                k, frame_id, scale = key
                item[key] = self._to_tensor(value)
                # if self.do_augmentation:
                item[(f'{k}_aug', frame_id, scale)] = self._to_tensor(augment(value))
            elif 'camera_matrix' in key or 'inv_camera_matrix' in key:
                item[key] = torch.from_numpy(value)
            elif 'depth' in key:
                item[key] = self._to_tensor(value) / 100  # Convert cm to m
            elif 'relative_pose' in key or 'absolute_pose' in key:
                item[key] = self._to_tensor(value)
            elif 'mask' in key:
                item[key] = torch.round(self._to_tensor(value))


# =========================================================


def extract_raw_data(
    raw_dataset_path: Path,
    odometry_dataset_path: Path,
    oxts: bool = True,
    gt_depth: bool = False,
) -> None:
    # yapf: disable
    # Mapping between KITTI Raw Drives and KITTI Odometry Sequences
    KITTI_RAW_SEQ_MAPPING = {
        0: {'date': '2011_10_03', 'drive': 27, 'start_frame': 0, 'end_frame': 4540},
        1: {'date': '2011_10_03', 'drive': 42, 'start_frame': 0, 'end_frame': 1100},
        2: {'date': '2011_10_03', 'drive': 34, 'start_frame': 0, 'end_frame': 4660},
        # 3:  {'date': '2011_09_26', 'drive': 67, 'start_frame':    0, 'end_frame':  800}, # No IMU
        4: {'date': '2011_09_30', 'drive': 16, 'start_frame': 0, 'end_frame': 270},
        5: {'date': '2011_09_30', 'drive': 18, 'start_frame': 0, 'end_frame': 2760},
        6: {'date': '2011_09_30', 'drive': 20, 'start_frame': 0, 'end_frame': 1100},
        7: {'date': '2011_09_30', 'drive': 27, 'start_frame': 0, 'end_frame': 1100},
        8: {'date': '2011_09_30', 'drive': 28, 'start_frame': 1100, 'end_frame': 5170},
        9: {'date': '2011_09_30', 'drive': 33, 'start_frame': 0, 'end_frame': 1590},
        10: {'date': '2011_09_30', 'drive': 34, 'start_frame': 0, 'end_frame': 1200},
    }
    # yapf: enable

    total_frames = 0
    for mapping in KITTI_RAW_SEQ_MAPPING.values():
        total_frames += (mapping['end_frame'] - mapping['start_frame'] + 1)

    if gt_depth:
        with tqdm(desc='Copying depth files', total=total_frames * 2, unit='files') as pbar:
            for sequence, mapping in KITTI_RAW_SEQ_MAPPING.items():
                # This is the improved gt depth using 5 consecutive frames from
                # "Sparsity invariant CNNs", J. Uhrig et al., 3DV, 2017.
                odometry_sequence_path = odometry_dataset_path / f'{sequence:02}' / 'gt_depth'
                split = 'val' if sequence == 4 else 'train'
                raw_sequence_path = raw_dataset_path / split / \
                                    f'{mapping["date"]}_drive_{mapping["drive"]:04}_sync' / \
                                    'proj_depth' / 'groundtruth'
                if not raw_sequence_path.exists():
                    continue
                for image in ['image_02', 'image_03']:
                    image_raw_sequence_path = raw_sequence_path / image
                    (odometry_sequence_path / image).mkdir(exist_ok=True, parents=True)
                    raw_filenames = sorted(image_raw_sequence_path.glob('*'))
                    for raw_filename in raw_filenames:
                        odometry_filename = odometry_sequence_path / image / raw_filename.name
                        frame = int(raw_filename.stem)
                        if mapping['start_frame'] <= frame <= mapping['end_frame']:
                            copyfile(raw_filename, odometry_filename)
                            pbar.update(1)
                            pbar.set_postfix({'sequence': sequence})

    if oxts:
        with tqdm(desc='Copying OXTS files', total=total_frames, unit='files') as pbar:
            for sequence, mapping in KITTI_RAW_SEQ_MAPPING.items():
                odometry_sequence_path = odometry_dataset_path / f'{sequence:02}' / 'oxts'
                raw_sequence_path = raw_dataset_path / \
                                    f'{mapping["date"]}' / \
                                    f'{mapping["date"]}_drive_{mapping["drive"]:04}_sync' / \
                                    'oxts'
                if not raw_sequence_path.exists():
                    continue
                odometry_sequence_path.mkdir(exist_ok=True, parents=True)
                copyfile(raw_sequence_path / 'dataformat.txt',
                         odometry_sequence_path / 'dataformat.txt')
                with open(raw_sequence_path / 'timestamps.txt', 'r', encoding='utf-8') as f:
                    timestamps = f.readlines()[mapping['start_frame']:mapping['end_frame'] + 1]
                with open(odometry_sequence_path / 'timestamps.txt', 'w', encoding='utf-8') as f:
                    f.writelines(timestamps)

                for image in ['data']:
                    image_raw_sequence_path = raw_sequence_path / image
                    (odometry_sequence_path / image).mkdir(exist_ok=True, parents=True)
                    raw_filenames = sorted(image_raw_sequence_path.glob('*'))
                    for raw_filename in raw_filenames:
                        odometry_filename = odometry_sequence_path / image / raw_filename.name
                        frame = int(raw_filename.stem)
                        if mapping['start_frame'] <= frame <= mapping['end_frame']:
                            copyfile(raw_filename, odometry_filename)
                            pbar.update(1)
                            pbar.set_postfix({'sequence': sequence})


# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', type=str)
    parser.add_argument('odom_path', type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--oxts', action='store_true')
    group.add_argument('--depth', action='store_true')
    args = parser.parse_args()

    extract_raw_data(Path(args.raw_path), Path(args.odom_path), oxts=args.oxts, gt_depth=args.depth)

    # seq = [i for i in range(11) if i != 3]
    # dataset = Kitti('/home/voedisch/data/kitti/odometry/dataset',
    #                 seq, (0, -1, 1), (0, 1, 2, 3),
    #                 192,
    #                 640,
    #                 with_mask=True,
    #                 poses=True)
    # print(dataset[0].keys())
