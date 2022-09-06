import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from datasets.utils import get_random_color_jitter


class ReplayBuffer(TorchDataset):
    def __init__(
        self,
        storage_dir: Path,
        dataset_type: str,
        state_path: Optional[Path] = None,
        height: int = 0,
        width: int = 0,
        scales: List[int] = None,
        frames: List[int] = None,
        num_workers: int = 1,
        do_augmentation: bool = False,
    ):
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = storage_dir
        self.dataset_type = dataset_type.lower()
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation

        # Restrict size of the replay buffer
        self.NUMBER_SAMPLES_PER_ENVIRONMENT = 100
        self.valid_indices = {}

        self.buffer_filenames = {}
        self.online_filenames = []
        if state_path is not None:
            self.load_state(state_path)

        # Precompute the resize functions for each scale relative to the previous scale
        # If scales is None, the size of the raw data will be used
        self.scales = scales
        self.frames = frames
        self.resize = {}
        if self.scales is not None:
            for s in self.scales:
                exp_scale = 2**s
                self.resize[s] = transforms.Resize(
                    (height // exp_scale, width // exp_scale),
                    interpolation=transforms.InterpolationMode.LANCZOS)

        # Ensure repeatability of experiments
        random.seed(42)

    def add(self, sample: Dict[str, Any], sample_filenames: Dict[str, Any]):
        index = sample['index'].item()
        assert index == sample_filenames['index']
        filename = self.storage_dir / f'{self.dataset_type}_{index:>05}.pkl'
        data = {
            key: value
            for key, value in sample.items() if 'index' in key or 'camera_matrix' in key
            or 'inv_camera_matrix' in key or 'relative_distance' in key
        }
        data['rgb', -1] = sample_filenames['images'][0]
        data['rgb', 0] = sample_filenames['images'][1]
        data['rgb', 1] = sample_filenames['images'][2]
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        self.online_filenames.append(filename)

    def get(self) -> Dict[str, Any]:
        return_data = {}
        for dataset, filenames in self.buffer_filenames.items():
            if dataset == self.dataset_type:
                continue
            # index = random.randint(0, len(filenames) - 1)
            index = random.sample(self.valid_indices[dataset], 1)[0]
            filename = filenames[index]
            data = self._get(filename)
            if not return_data:
                return_data = data
            else:
                for key in return_data:
                    return_data[key] = torch.cat([return_data[key], data[key]])
        return return_data

    def save_state(self):
        filename = self.storage_dir / 'buffer_state.pkl'
        buffer_filenames = self.buffer_filenames
        if self.dataset_type in buffer_filenames.keys():
            buffer_filenames[self.dataset_type] += self.online_filenames
        else:
            buffer_filenames[self.dataset_type] = self.online_filenames
        data = {'filenames': buffer_filenames}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved reply buffer state to: {filename}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

    def load_state(self, state_path: Path):
        with open(state_path, 'rb') as f:
            data = pickle.load(f)
            self.buffer_filenames = data['filenames']
        print(f'Load replay buffer state from: {state_path}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

        for key, value in self.buffer_filenames.items():
            random.seed(42)
            center_sequences = sorted(random.sample(range(len(value)),
                                                    self.NUMBER_SAMPLES_PER_ENVIRONMENT))
            assert center_sequences[0] >= 1 and center_sequences[-1] <= len(value)-2
            self.valid_indices[key] = center_sequences
            print(f'{key + ":":<12} {len(center_sequences):>5}')

    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        return self.get()

    def __len__(self):
        return 1000000  # Fixed number as the sampling is handled in the get() function

    def _get(self, filename, include_batch=True):
        if self.do_augmentation:
            color_augmentation = get_random_color_jitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2),
                                                         (-.1, .1))

        with open(filename, 'rb') as f:
            data = pickle.load(f)
        for frame in self.frames:
            rgb = Image.open(data['rgb', frame]).convert('RGB')
            rgb = self.resize[0](rgb)
            data['rgb', frame, 0] = rgb
            for scale in self.scales:
                if scale == 0:
                    continue
                data['rgb', frame, scale] = self.resize[scale](data['rgb', frame, scale - 1])
            for scale in self.scales:
                data['rgb', frame, scale] = transforms.ToTensor()(data['rgb', frame, scale])
                if include_batch:
                    data['rgb', frame, scale] = data['rgb', frame, scale].unsqueeze(0)
                if self.do_augmentation:
                    data['rgb_aug', frame, scale] = color_augmentation(data['rgb', frame, scale])
                else:
                    data['rgb_aug', frame, scale] = data['rgb', frame, scale]
            del data['rgb', frame]  # Removes the filename string
        if not include_batch:
            for key in data:
                if not ('rgb' in key or 'rgb_aug' in key):
                    data[key] = data[key].squeeze(0)
        return data
