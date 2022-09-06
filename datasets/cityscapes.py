import json
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from datasets.utils import Dataset


class Cityscapes(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path, PathLike],
        subsets: Union[str, List[str], Tuple[str, ...]],
        frame_ids: Union[List[int], Tuple[int, ...]],
        scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        do_augmentation: bool = False,
        views: Union[List[str], Tuple[str, ...]] = ('left', ),
        with_depth: bool = False,
        with_mask: bool = False,
    ) -> None:
        if any(v != 'left' for v in views):
            raise ValueError('Cityscapes supports only views = ["left"]')
        super().__init__(data_path, frame_ids, scales, height, width, do_augmentation, views,
                         with_depth, with_mask)

        # The following subsets are available:
        # -) train --> used for pretraining
        # -) val --> used for validation during training
        # -) test --> not used in the published version
        # -) frankfurt --> not used in the published version. Refers to "allFrames_frankfurt"
        subsets = (subsets, ) if isinstance(subsets, str) else subsets
        if any(s not in ['train', 'val', 'test', 'frankfurt'] for s in subsets):
            raise ValueError('subsets must be one of ["train", "val", "test"]')
        if 'frankfurt' in subsets and len(subsets) > 1:
            raise ValueError('The subset "frankfurt" cannot be combined with other subsets.')
        self.subsets = subsets

        self._load_image_filenames()
        self.vehicle_filenames = self._get_filenames(mode='vehicle')
        self.timestamp_filenames = self._get_filenames(mode='timestamp')
        self.disparity_filenames = self._get_filenames(mode='disparity') if self.with_depth else []
        if self.with_mask:
            self._load_mask_filenames()

    def _get_filenames(self, mode: str) -> List[Path]:
        if self.subsets[0] == 'frankfurt':
            valid_modes = ['rgb_left', 'vehicle', 'timestamp']
            if mode not in valid_modes:
                raise ValueError(f'mode must be one of {valid_modes}. [subset == "frankfurt"]')
            mode_dir = {
                'rgb_left': 'leftImg8bit_allFrames',
                'vehicle': 'vehicle_allFrames',
                'timestamp': 'timestamp_allFrames',
            }[mode]
            subsets = ('val', )
        else:
            valid_modes = ['rgb_left', 'vehicle', 'timestamp', 'disparity', 'mask_left']
            if mode not in valid_modes:
                raise ValueError(f'mode must be one of {valid_modes}.')
            mode_dir = {
                'rgb_left': 'leftImg8bit_sequence',
                'vehicle': 'vehicle_sequence',
                'timestamp': 'timestamp_sequence',
                'disparity': 'disparity_sequence',
                'mask_left': 'segm_mask_sequence',
            }[mode]
            subsets = self.subsets
        mode_ext = {
            'rgb_left': 'png',
            'vehicle': 'json',
            'timestamp': 'txt',
            'disparity': 'png',
            'mask_left': 'png',
        }[mode]
        filenames = []
        counter_indices = 0
        for subset in subsets:
            subset_data_path = self.data_path / mode_dir / subset
            cities = sorted(subset_data_path.glob('*'))
            for city_path in cities:
                city_filenames = sorted(city_path.glob(f'*.{mode_ext}'))
                filenames += city_filenames
                city_sequences = self._divide_into_sequences(city_filenames)
                for city_sequence, count in city_sequences.items():
                    if city_sequence not in self.sequence_indices:
                        self.sequence_indices[city_sequence] = (counter_indices,
                                                                counter_indices + count - 1)
                        counter_indices += count
        return filenames

    @staticmethod
    def _divide_into_sequences(city_filenames: List[Path]) -> Dict[str, int]:
        # filename: <city>_<seq>_<cnt>_<type>.<ext>
        # Both <seq> and <cnt> are used in this function to detect the start of a new sequence.
        filenames = [f.stem for f in city_filenames]
        city = city_filenames[0].stem.split('_')[0]
        city_sequences = {}
        sequence_length = 1
        sequence_counter = 0
        for file_1, file_2 in zip(filenames, filenames[1:]):
            seq_1, seq_2 = int(file_1.split('_')[1]), int(file_2.split('_')[1])
            cnt_1, cnt_2 = int(file_1.split('_')[2]), int(file_2.split('_')[2])
            if seq_1 == seq_2 and cnt_1 + 1 == cnt_2:
                # Continuation of the same sequence
                sequence_length += 1
            elif (seq_1 != seq_2) or (seq_1 == seq_2 and cnt_1 + 1 != cnt_2):
                # New sequence started. Either increment in seq or non-consecutive cnt.
                city_sequences[f'{city}_{sequence_counter:06}'] = sequence_length
                sequence_length = 1
                sequence_counter += 1
            else:
                raise RuntimeError('Ran into an unexpected situation.')
        # Add the final sequence
        city_sequences[f'{city}_{sequence_counter:06}'] = sequence_length
        return city_sequences

    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        img_filenames, mask_filenames, index, do_color_augmentation, do_flip = self._pre_getitem(
            index)

        # load the color images
        item = {('index'): index}
        original_image_shape = None
        for frame_id in self.frame_ids:
            rgb = Image.open(img_filenames[index + frame_id]).convert('RGB')
            if frame_id == 0:
                original_image_shape = rgb.size
            if do_flip:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            if len(self.scales) > 0:
                rgb = self.resize[0](rgb)
                item[('rgb', frame_id, 0)] = rgb
            else:
                item[('rgb', frame_id, -1)] = rgb

        # Adjusting intrinsics to match each scale in the pyramid
        normalized_camera_matrix, baseline = self._load_camera_calibration(
            index, original_image_shape)
        for scale in range(len(self.scales)):
            camera_matrix, inv_camera_matrix = self._scale_camera_matrix(
                normalized_camera_matrix, scale)
            item[('camera_matrix', scale)] = camera_matrix
            item[('inv_camera_matrix', scale)] = inv_camera_matrix
        if len(self.scales) == 0:
            # Load camera matrix of raw image data
            camera_matrix, baseline = self._load_camera_calibration(index, (1, 1))
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

        # Load relative distance (except for first frame)
        for frame_id in self.frame_ids[1:]:
            item[('relative_distance', frame_id)] = self._load_relative_distance(index + frame_id)

        # Load unscaled depth based on the provided disparity maps
        if self.with_depth:
            for frame_id in self.frame_ids:
                depth = self._load_depth(index + frame_id, normalized_camera_matrix, baseline,
                                         original_image_shape)
                if do_flip:
                    depth = np.fliplr(depth)
                item[('depth', frame_id, -1)] = depth

        self._post_getitem(item, do_color_augmentation)  # This will also call _preprocess()
        return item

    def _load_camera_calibration(
        self,
        index: int,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, float]:
        """
        Returns the intrinsics matrix normalized by the original image size.
        """
        width, height = image_shape
        city = self.vehicle_filenames[index].parent.name
        subset = self.vehicle_filenames[index].parents[1].name
        sequence = '_'.join(self.vehicle_filenames[index].name.split('_')[:2])
        data_path = self.data_path / 'camera' / subset / city
        # We assume that the camera intrinsics are constant for one recording
        camera_file = sorted(data_path.glob(f'{sequence}_*_camera.json'))[0]
        with open(camera_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        intrinsics = data['intrinsic']
        baseline = data['extrinsic']['baseline']
        camera_matrix = np.array(
            [[intrinsics['fx'], 0, intrinsics['u0'], 0], [0, intrinsics['fy'], intrinsics['v0'], 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32)
        camera_matrix[0, :] /= width
        camera_matrix[1, :] /= height
        return camera_matrix, baseline

    def _load_relative_distance(self, index: int) -> float:
        """
        Distance in meters and with respect to the previous frame.
        """
        previous_timestamp = np.loadtxt(str(self.timestamp_filenames[index - 1]))
        current_timestamp = np.loadtxt(str(self.timestamp_filenames[index]))
        delta_timestamp = (current_timestamp - previous_timestamp) / 1e9  # ns to s
        with open(self.vehicle_filenames[index - 1], 'r', encoding='utf-8') as f:
            previous_speed = json.load(f)['speed']
        with open(self.vehicle_filenames[index], 'r', encoding='utf-8') as f:
            current_speed = json.load(f)['speed']
        speed = (previous_speed + current_speed) / 2  # m/s
        distance = speed * delta_timestamp  # m
        return distance

    def _load_depth(
        self,
        index: int,
        scaled_camera_matrix: np.ndarray,
        baseline: float,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Depth in meters computed based on the provided disparity maps.
        Zero elements encode missing disparity values.
        """
        disparity = cv2.imread(str(self.disparity_filenames[index]),
                               cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256  # According to README
        # Reconstruct the value of the raw data
        focal_length_x = scaled_camera_matrix[0, 0] * image_shape[0]
        depth = np.zeros_like(disparity)
        depth[disparity > 0] = (baseline * focal_length_x) / disparity[disparity > 0]  # m
        return depth

    def _preprocess(
        self,
        item: Dict[Any, Any],
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
                item[(f'{k}_aug', frame_id, scale)] = self._to_tensor(augment(value))
            elif 'camera_matrix' in key or 'inv_camera_matrix' in key:
                item[key] = torch.from_numpy(value)
            elif 'depth' in key:
                item[key] = self._to_tensor(value)
            elif 'mask' in key:
                item[key] = torch.round(self._to_tensor(value))


# if __name__ == '__main__':
#     dataset = Cityscapes('/home/voedisch/data/cityscapes', ('frankfurt'), (0, -1, 1),
#                          (0, 1, 2, 3),
#                          192,
#                          640,
#                          with_mask=False,
#                          with_depth=True)
#     print(dataset[0].keys())
