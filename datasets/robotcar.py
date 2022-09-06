import argparse
import bisect
import multiprocessing as mp
import os
import re
from functools import partial
from math import sqrt
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from torch import Tensor
from tqdm import tqdm

from datasets.utils import Dataset


class Robotcar(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path, PathLike],
        sequences: Union[str, List[str], Tuple[str, ...]],
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
        every_n_frame: int = 1,
        start_frame: int = 750,
        end_frame: int = -1,
    ) -> None:
        if with_mask:
            raise ValueError('Robotcar does not support loading masks.')
        if with_depth:
            raise ValueError('Robotcar does not support loading depth.')

        # This is actually the center
        if any(v != 'left' for v in views):
            raise ValueError('Robotcar supports only views = ["left"]')
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

        self.include_poses = poses
        self.every_n_frame = every_n_frame
        self.start_frame = start_frame
        self.end_frame = end_frame

        sequences = (sequences, ) if isinstance(sequences, str) else sequences
        self.sequences = sequences

        self._load_image_filenames()
        self.timestamps = [int(f.stem) for f in self.left_img_filenames]
        self.camera_matrix = self._load_camera_calibration()
        self.velocity = self._load_velocity()
        self.relative_distances = self._compute_relative_distance()

        self.global_poses = None
        self.relative_poses = None
        if self.include_poses:
            self.global_poses = self._load_global_poses()

        # Filter data to meet minimum distance requirements
        if self.min_distance > 0:
            self._filter_by_distance(self.min_distance)

        if self.include_poses:
            self.relative_poses = self._load_relative_poses()  # Has to be done in the end

    def _get_filenames(self, mode: str) -> List[Path]:
        valid_modes = ['rgb_left']
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')
        mode_dir = {
            'rgb_left': 'stereo/center',
        }[mode]
        mode_ext = {
            'rgb_left': 'png',
        }[mode]
        filenames = []
        for sequence in self.sequences:
            sequence_index = [len(filenames)]
            sequence_data_path = self.data_path / sequence / mode_dir
            filenames += sorted(sequence_data_path.glob(
                f'*.{mode_ext}'))[self.start_frame:self.end_frame:self.every_n_frame]
            sequence_index.append(len(filenames) - 1)
            if sequence not in self.sequence_indices:
                self.sequence_indices[sequence] = tuple(sequence_index)
        return filenames

    def _load_velocity(self) -> List[float]:
        """
        Returns corresponding velocity for each image
        """
        speed = []
        for sequence in self.sequences:
            ins_file = self.data_path / sequence / 'gps' / 'ins.csv'
            column_list = ['timestamp', 'velocity_north', 'velocity_east', 'velocity_down']
            data = pd.read_csv(ins_file, usecols=column_list)
            raw_timestamps = data['timestamp'].to_numpy()
            raw_speed = np.linalg.norm(data.to_numpy()[:, 1:], axis=1)  # m/s
            # Linearly interpolate speed for each frame
            speed = interp1d(raw_timestamps, raw_speed)(self.timestamps)
        return speed

    def _load_camera_calibration(self) -> np.ndarray:
        """
        Returns the intrinsics matrix normalized by the original image size.
        """
        rgb = Image.open(self.left_img_filenames[0]).convert('RGB')
        width, height = rgb.size

        camera_file = self.data_path / 'camera_models' / 'stereo_narrow_left.txt'
        with open(camera_file, 'r', encoding='utf-8') as f:
            vals = [float(x) for x in next(f).split()]
            focal_length = (vals[0], vals[1])
            principal_point = (vals[2], vals[3])
        camera_matrix = np.array(
            [[focal_length[0], 0, principal_point[0], 0],
             [0, focal_length[1], principal_point[1], 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32)
        camera_matrix[0, :] /= width
        camera_matrix[1, :] /= height
        return camera_matrix

    def _load_global_poses(self) -> np.ndarray:
        """
        Reads the poses based on the provided RTK data
        """
        global_poses = []
        relative_poses = []
        for sequence in self.sequences:
            rtk_file = self.data_path / 'rtk' / sequence / 'rtk.csv'
            column_list = ['timestamp', 'northing', 'easting', 'down', 'roll', 'pitch', 'yaw']
            data = pd.read_csv(rtk_file, usecols=column_list)
            timestamps = data['timestamp'].to_numpy()
            utm = data.to_numpy()[:, 1:4]
            rpy = data.to_numpy()[:, 4:]  # pylint: disable=no-member
            utm -= utm[0, :]  # Set first pose to origin (0, 0, 0) to avoid large numbers
            utm[:, [1, 2]] = utm[:, [2, 1]]  # Swap y and z axes
            rpy[:, [1, 2]] = rpy[:, [2, 1]]
            utm[:, 2] *= -1
            sequence_poses = list(_xyzrpy_to_tmat(utm, rpy).astype(dtype=np.float32))
            sequence_poses = _interpolate_poses(timestamps, sequence_poses, self.timestamps,
                                                self.timestamps[0])
            sequence_poses = np.stack(sequence_poses)
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
        self.velocity = [self.velocity[i] for i in keep_indices]
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

        # Load relative distance (except for first frame)
        for frame_id in self.frame_ids[1:]:
            item['relative_distance', frame_id] = self.relative_distances[index + frame_id]

        if self.include_poses:
            # Load the poses (except for the first frame)
            for frame_id in self.frame_ids[1:]:
                item['relative_pose', frame_id] = self.relative_poses[index + frame_id]
                if do_flip:
                    # We also have to flip the relative pose (rotate around y-axis)
                    item[('relative_pose', 0)][2, 0] *= -1
                    item[('relative_pose', 0)][0, 2] *= -1
                item[('absolute_pose', frame_id)] = self.global_poses[index + frame_id]

        self._post_getitem(item, do_color_augmentation)  # This will also call _preprocess()
        return item

    def _load_relative_distance(self, index: int) -> float:
        """
        Distance in meters and with respect to the previous frame
        """
        previous_timestamp = self.timestamps[index - 1]
        current_timestamp = self.timestamps[index]
        delta_timestamp = (current_timestamp - previous_timestamp) / 1e6  # ms to s
        speed = self.velocity[index]  # m/s
        distance = speed * delta_timestamp  # m
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
            elif 'relative_pose' in key or 'absolute_pose' in key:
                item[key] = self._to_tensor(value)


# =========================================================


def _xyzrpy_to_tmat(utm: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    assert utm.shape == rpy.shape

    poses = np.array([np.eye(4)] * utm.shape[0])
    poses[:, :3, :3] = Rotation.from_euler('zyx', rpy).as_matrix()
    poses[:, :3, 3] = utm
    return poses


# Adapted from:
# https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/interpolate_poses.py
def _interpolate_poses(pose_timestamps,
                       abs_poses,
                       requested_timestamps_,
                       origin_timestamp,
                       absolute_poses=False):
    """Interpolate between absolute poses.
    Args:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the
        timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to
        this frame.
    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each
        requested timestamp.
    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order
    """
    requested_timestamps = requested_timestamps_.copy()
    requested_timestamps.insert(0, origin_timestamp)
    requested_timestamps = np.array(requested_timestamps)
    pose_timestamps = np.array(pose_timestamps)

    if len(pose_timestamps) != len(abs_poses):
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i - 1] >= pose_timestamps[i]:
            raise ValueError('Pose timestamps must be in ascending order')

        abs_quaternions[:, i] = _so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = np.ravel(pose[0:3, 3])

    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps]
    if max(upper_indices) >= len(pose_timestamps):
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]
    lower_indices = [u - 1 for u in upper_indices]

    fractions = (requested_timestamps - pose_timestamps[lower_indices]) / \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = np.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    if not absolute_poses:
        poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat)

    poses_out = [np.empty(0)] * (len(requested_timestamps) - 1)
    for i in range(1, len(requested_timestamps)):
        poses_out[i - 1] = np.asarray(poses_mat[0:4, i * 4:(i + 1) * 4])

    return poses_out


# Adapted from:
# https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/transform.py
def _so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion
    Args:
        so3: 3x3 rotation matrix
    Returns:
        numpy.ndarray: quaternion [w, x, y, z]
    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError('SO3 matrix must be 3x3')

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except ValueError:
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


# =========================================================


# Use this function to save undistorted copies of the raw images provided in RobotCar
def undistort_images(data_path_in: str, models_path: str) -> None:
    data_path_out = data_path_in
    data_path_in = data_path_in.rstrip('/') + '_distorted'
    os.rename(data_path_out, data_path_in)
    Path(data_path_out).mkdir(parents=True, exist_ok=True)

    model = CameraModel(models_path, data_path_in)
    image_files = sorted(x for x in Path(data_path_in).glob('*.png'))
    # The other photos are overexposed or taken when the car was not (yet) on the road
    image_files = image_files[1112:-147]

    num_workers = mp.cpu_count() - 1
    with tqdm(total=len(image_files)) as pbar:
        with mp.Pool(num_workers) as pool:
            for _ in pool.imap_unordered(
                    partial(_undistort, data_path_out=data_path_out, model=model), image_files):
                pbar.update(1)


def _undistort(image_file: Path, data_path_out: str, model):
    new_image_file = Path(data_path_out) / image_file.name
    if not new_image_file.exists():
        image = Image.fromarray(_load_image(str(image_file), model))
        image.save(new_image_file)


# Adapted from:
# https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/image.py
def _load_image(image_path, model=None, debayer=True):
    """Loads and rectifies an image from file.
    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.
    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image
    """
    BAYER_STEREO = 'gbrg'
    BAYER_MONO = 'rggb'

    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    if debayer:
        img = demosaic(img, pattern)
    if model:
        img = model.undistort(img)

    return np.array(img).astype(np.uint8)


# Adapted from:
# https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/camera_model.py
class CameraModel:
    """Provides intrinsic parameters and undistortion LUT for a camera.
    Attributes:
        camera (str): Name of the camera.
        camera sensor (str): Name of the sensor on the camera for multi-sensor cameras.
        focal_length (tuple[float]): Focal length of the camera in horizontal and vertical axis,
        in pixels.
        principal_point (tuple[float]): Principal point of camera for pinhole projection model,
        in pixels.
        G_camera_image (:obj: `numpy.matrixlib.defmatrix.matrix`): Transform from image frame to
        camera frame.
        bilinear_lut (:obj: `numpy.ndarray`): Look-up table for undistortion of images, mapping
        pixels in an undistorted
            image to pixels in the distorted image
    """
    def __init__(self, models_dir, images_dir):
        """Loads a camera model from disk.
        Args:
            models_dir (str): directory containing camera model files.
            images_dir (str): directory containing images for which to read camera model.
        """
        self.camera = None
        self.camera_sensor = None
        self.focal_length = None
        self.principal_point = None
        self.G_camera_image = None
        self.bilinear_lut = None

        self.__load_intrinsics(models_dir, images_dir)
        self.__load_lut(models_dir, images_dir)

    def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.
        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative
            to camera frame.
            image_size (tuple[int]): dimensions of image in pixels
        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates
            of a point in pixels.
            numpy.array: array of depth values for points in image.
        Note:
            Number of output points m will be less than or equal to number of input points n, as
            points that do not
            project into the image are discarded.
        """
        if xyz.shape[0] == 3:
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(self.G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        in_img = [
            i for i in range(0, uv.shape[1])
            if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]
        ]

        return uv[:, in_img], np.ravel(xyzw[2, in_img])

    def undistort(self, image):
        """Undistorts an image.
        Args:
            image (:obj: `numpy.ndarray`): A distorted image. Must be demosaiced - ie. must be a
            3-channel RGB image.
        Returns:
            numpy.ndarray: Undistorted version of image.
        Raises:
            ValueError: if image size does not match camera model.
            ValueError: if image only has a single channel.
        """
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]:
            raise ValueError('Incorrect image size for camera model')

        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))

        if len(image.shape) == 1:
            raise ValueError('Undistortion function only works with multi-channel images')

        undistorted = np.rollaxis(
            np.array([
                map_coordinates(image[:, :, channel], lut, order=1)
                for channel in range(0, image.shape[2])
            ]), 0, 3)

        return undistorted.astype(image.dtype)

    def __get_model_name(self, images_dir):
        self.camera = re.search('(stereo|mono_(left|right|rear))', images_dir).group(0)
        if self.camera == 'stereo':
            self.camera_sensor = re.search('(left|center_distorted|centre_distorted|right)',
                                           images_dir).group(0)
            if self.camera_sensor == 'left':
                return 'stereo_wide_left'
            if self.camera_sensor == 'right':
                return 'stereo_wide_right'
            if self.camera_sensor in ['center_distorted', 'centre_distorted']:
                return 'stereo_narrow_left'
            raise RuntimeError('Unknown camera model for given directory: ' + images_dir)
        return self.camera

    def __load_intrinsics(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        intrinsics_path = os.path.join(models_dir, model_name + '.txt')

        with open(intrinsics_path, 'r', encoding='utf-8') as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            self.focal_length = (vals[0], vals[1])
            self.principal_point = (vals[2], vals[3])

            G_camera_image = []
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
            self.G_camera_image = np.array(G_camera_image)

    def __load_lut(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        lut_path = os.path.join(models_dir, model_name + '_distortion_lut.bin')

        lut = np.fromfile(lut_path, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.bilinear_lut = lut.transpose()


# =========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    parser.add_argument('models_path', type=str)
    args = parser.parse_args()

    undistort_images(args.img_path, args.models_path)

    # dataset = Robotcar('/home/voedisch/data/robotcar',
    #                    '2015-08-12-15-04-18', (0, -1, 1), (0, 1, 2, 3),
    #                    192,
    #                    640,
    #                    poses=True)
    # print(dataset[0].keys())
