import pickle
import random
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from torchvision.transforms import Compose, Lambda


class Dataset(TorchDataset):
    def __init__(
        self,
        data_path: Union[str, Path, PathLike],
        frame_ids: Union[List[int], Tuple[int, ...]],
        scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        do_augmentation: bool = False,
        views: Union[List[str], Tuple[str, ...]] = ('left', ),
        with_depth: bool = False,
        with_mask: bool = False,
        use_cache: bool = False,
        min_distance: float = 0,
    ) -> None:
        super().__init__()

        if any(v not in ['left', 'right'] for v in views):
            raise ValueError('views must be one of ["left", "right"]')
        if sum(x is None for x in [scales, height, width]) == 1:
            raise ValueError('Either none or all of [scales, height, width] must be omitted.')

        self.data_path = Path(data_path)
        self.frame_ids = sorted(frame_ids)
        self.scales = scales if scales is not None else ()
        self.height = height
        self.width = width
        self.do_augmentation = do_augmentation
        self.views = tuple(set(views))
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.min_distance = min_distance  # Minimum distance between poses [in meters]

        # If loading from the original source takes some time, we cache the extracted data
        self.use_cache = use_cache
        self.cache_file = None

        # Data augmentation
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        # Precompute the resize functions for each scale relative to the previous scale
        # If scales is None, the size of the raw data will be used
        self.resize = {}
        for s in self.scales:
            exp_scale = 2**s
            self.resize[s] = transforms.Resize((self.height // exp_scale, self.width // exp_scale),
                                               interpolation=transforms.InterpolationMode.LANCZOS)

        self.sequence_indices = {}
        self.left_img_filenames = []
        self.right_img_filenames = []
        self.left_mask_filenames = []
        self.right_mask_filenames = []

    def _load_image_filenames(self) -> None:
        if 'left' in self.views:
            self.left_img_filenames = self._get_filenames(mode='rgb_left')
        if 'right' in self.views:
            self.right_img_filenames = self._get_filenames(mode='rgb_right')
        if len(self.views) == 2:
            assert len(self.left_img_filenames) == len(self.right_img_filenames)

    def _load_mask_filenames(self) -> None:
        if 'left' in self.views:
            self.left_mask_filenames = self._get_filenames(mode='mask_left')
        if 'right' in self.views:
            self.right_mask_filenames = self._get_filenames(mode='mask_right')
        if len(self.views) == 2:
            assert len(self.left_mask_filenames) == len(self.right_mask_filenames)

    def _len_frames(self) -> int:
        """
        Subtract requested neighboring frames on either side.
        """
        if 'left' in self.views:
            return len(self.left_img_filenames) - 2 * len(self.sequence_indices)
        return len(self.right_img_filenames) - 2 * len(self.sequence_indices)

    def __len__(self) -> int:
        """
        Multiply by number of views to account for left and right images.
        """
        return len(self.views) * self._len_frames()

    def _scale_camera_matrix(self, camera_matrix: np.ndarray,
                             scale: int) -> Tuple[np.ndarray, np.ndarray]:
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[0, :] *= self.width // (2**scale)
        scaled_camera_matrix[1, :] *= self.height // (2**scale)
        inv_scaled_camera_matrix = np.linalg.pinv(scaled_camera_matrix)
        return scaled_camera_matrix, inv_scaled_camera_matrix

    def _pre_getitem(self, index: int) -> Tuple[List[Path], List[Path], int, bool, bool]:

        if index < 0 or index >= self.__len__():
            raise IndexError()

        if len(self.views) == 2:
            if index < self._len_frames():
                img_filenames = self.left_img_filenames
            else:
                img_filenames = self.right_img_filenames
                index -= self._len_frames()
        elif self.views[0] == 'left':
            img_filenames = self.left_img_filenames
        else:
            img_filenames = self.right_img_filenames

        mask_filenames = []
        if self.with_mask:
            if len(self.views) == 2:
                if index < self._len_frames():
                    mask_filenames = self.left_mask_filenames
                else:
                    mask_filenames = self.right_mask_filenames
                    index -= self._len_frames()
            elif self.views[0] == 'left':
                mask_filenames = self.left_mask_filenames
            else:
                mask_filenames = self.right_mask_filenames

        # Get number of shift indices
        # The formula is: index + 2*i + 1, where i is the i-th element of the ordered sequences
        for i, seq_indices in enumerate(self.sequence_indices.values()):
            if seq_indices[0] < index + 2 * i + 1 < seq_indices[1]:
                index += 2 * i + 1
                break

        # Determine whether to apply data augmentation
        do_color_augmentation = self.do_augmentation and random.random() > .5
        do_flip = self.do_augmentation and random.random() > .5

        return img_filenames, mask_filenames, index, do_color_augmentation, do_flip

    def _post_getitem(self, item: Dict[Any, Any], do_color_augmentation: bool) -> None:
        # Resize images
        # Convert to list object as we are changing the size during the iteration
        for key in list(item.keys()):
            if 'rgb' in key or 'mask' in key:
                k, frame_id, _ = key
                for scale in self.scales:
                    if scale == 0:
                        continue
                    item[(k, frame_id, scale)] = self.resize[scale](item[(k, frame_id, scale - 1)])

        # Apply color augmentation
        if do_color_augmentation:
            color_augmentation = get_random_color_jitter(self.brightness, self.contrast,
                                                         self.saturation, self.hue)
            self._preprocess(item, color_augmentation)
        else:
            self._preprocess(item, augment=(lambda x: x))

    def _load_from_cache(self, cache_name: str) -> Union[None, Any]:
        if self.cache_file is None:
            raise RuntimeError('cache_file not set for this dataset.')
        # Separate cache files for the different names to avoid reading a large cache file
        cache_file = self.cache_file.parent / \
                     f'{self.cache_file.stem}_{cache_name}{self.cache_file.suffix}'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        return None

    def _save_to_cache(self, cache_name: str, data: Any, replace: bool = False) -> None:
        if self.cache_file is None:
            raise RuntimeError('cache_file not set for this dataset.')
        # Separate cache files for the different names to avoid reading a large cache file
        cache_file = self.cache_file.parent / \
                     f'{self.cache_file.stem}_{cache_name}{self.cache_file.suffix}'
        # Write new file or replace existing
        if not cache_file.exists() or replace:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def _get_filenames(self, mode: str) -> List[Path]:
        raise NotImplementedError

    @abstractmethod
    def _preprocess(
        self,
        item: Dict[Any, Any],
        augment: Union[Callable, Tuple[Tensor, Optional[float], Optional[float], Optional[float],
                                       Optional[float]]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        raise NotImplementedError

    @staticmethod
    def _to_tensor(data) -> Tensor:
        return transforms.ToTensor()(data)

    def get_item_filenames(self, index: int):
        all_img_filenames, all_mask_filenames, index, _, _ = self._pre_getitem(index)
        img_filenames = []
        mask_filenames = []
        for frame_id in self.frame_ids:
            img_filenames.append(all_img_filenames[index + frame_id])
            if all_mask_filenames:
                mask_filenames.append(all_mask_filenames[index + frame_id])
        filenames = {
            'index': index,
            'images': img_filenames,
            'masks': mask_filenames,
        }
        return filenames


# =============================================================================
# Adapted from:
# https://github.com/pytorch/vision/pull/3001#issuecomment-814919958
def get_random_color_jitter(
    brightness: Optional[Tuple[float, float]] = None,
    contrast: Optional[Tuple[float, float]] = None,
    saturation: Optional[Tuple[float, float]] = None,
    hue: Optional[Tuple[float, float]] = None,
) -> Compose:
    transforms_ = []

    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        transforms_.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        transforms_.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        transforms_.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        transforms_.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

    random.shuffle(transforms_)
    transforms_ = Compose(transforms_)
    return transforms_


# =============================================================================


def augment_data(sample):
    # Data augmentation
    brightness = (0.8, 1.2)
    contrast = (0.8, 1.2)
    saturation = (0.8, 1.2)
    hue = (-0.1, 0.1)

    augmentation = get_random_color_jitter(brightness, contrast, saturation, hue)

    augmented_sample = {}
    for key, value in sample.items():
        if 'rgb_aug' in key:
            augmented_sample[key] = transforms.ToTensor()(augmentation(transforms.ToPILImage()(
                value[0, ...]).convert('RGB'))).unsqueeze(0)
        else:
            augmented_sample[key] = value.detach().clone()
    return augmented_sample


# =============================================================================


def show_images(batch, scales=(0, 1, 2, 3), frames=(-1, 0, 1), augmented=False):
    batch_size = batch['index'].shape[0]
    num_frames = len(frames)

    for s in scales:
        fig, axs = plt.subplots(nrows=batch_size,
                                ncols=num_frames,
                                figsize=(10, 18 / 15 * batch_size))
        axs = axs.reshape(batch_size, num_frames)
        for b in range(batch_size):
            for f in frames:
                if augmented:
                    axs[b, f + 1].imshow(batch['rgb_aug', f, s][b, :, :, :].permute(1, 2, 0))
                else:
                    axs[b, f + 1].imshow(batch['rgb', f, s][b, :, :, :].permute(1, 2, 0))
                axs[b, f + 1].axis('off')
                if f != -1:
                    axs[b, f + 1].set_title(f'{batch["relative_distance", f][b]:.2f}')
        fig.tight_layout()
        fig.show()
        plt.close(fig)
