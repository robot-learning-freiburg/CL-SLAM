import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from datasets.utils import get_random_color_jitter
from loop_closure_detection.encoder import FeatureEncoder


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
            batch_size: int = 1,
            maximize_diversity: bool = False,
            max_buffer_size: int = np.iinfo(int).max,
            similarity_threshold: float = 1,
            similarity_sampling: bool = True,
    ):
        self.storage_dir = storage_dir
        # self._reset_storage_dir()

        self.dataset_type = dataset_type.lower()
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation
        self.batch_size = batch_size

        # Restrict size of the replay buffer
        self.NUMBER_SAMPLES_PER_ENVIRONMENT = 100
        self.valid_indices = {}

        self.buffer_filenames = {}
        self.online_filenames = []

        # Precompute the resize functions for each scale relative to the previous scale
        # If scales is None, the size of the raw data will be used
        self.scales = scales
        self.frames = frames
        self.resize = {}
        if self.scales is not None:
            for s in self.scales:
                exp_scale = 2 ** s
                self.resize[s] = transforms.Resize(
                    (height // exp_scale, width // exp_scale),
                    interpolation=transforms.InterpolationMode.LANCZOS)

        # Ensure repeatability of experiments
        self.target_sampler = np.random.default_rng(seed=42)

        # Dissimilarity-based buffer
        self.similarity_sampling = similarity_sampling
        self.maximize_diversity = maximize_diversity
        self.buffer_size = max_buffer_size
        self.similarity_threshold = similarity_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_encoder = FeatureEncoder(self.device)
        self.faiss_index = None
        self.faiss_index_offset = 0
        self.distance_matrix = None
        self.distance_matrix_indices = None

        if state_path is not None:
            self.load_state(state_path)

    def add(self, sample: Dict[str, Any], sample_filenames: Dict[str, Any],
            image_features: Optional[Tensor] = None, verbose: bool = False):
        # pylint: disable=no-value-for-parameter
        index = sample['index'].item()
        assert index == sample_filenames['index']

        index += self.faiss_index_offset

        if self.faiss_index is None:
            if image_features is None:
                num_features = self.feature_encoder.num_features
            else:
                num_features = image_features.shape[1]
            self.faiss_index = faiss.IndexIDMap(
                faiss.index_factory(num_features, 'Flat', faiss.METRIC_INNER_PRODUCT))

        if image_features is None:
            image_features = self.feature_encoder(sample['rgb', 0, 0]).detach().cpu().numpy()
        faiss.normalize_L2(image_features)  # The inner product becomes cosine similarity

        add_sample = False
        remove_sample = None
        if self.maximize_diversity:

            # Only add if sufficiently dissimilar to the existing samples
            if self.faiss_index.ntotal == 0:
                similarity = 0
            else:
                similarity = self.faiss_index.search(image_features, 1)[0][0][0]

            if similarity < self.similarity_threshold:
                self.faiss_index.add_with_ids(image_features, np.array([index]))
                add_sample = True
                if verbose:
                    print(f'Added sample {index} to the replay buffer | similarity {similarity}')

                if self.faiss_index.ntotal > self.buffer_size:
                    # Maximize the diversity in the replay buffer
                    if self.distance_matrix is None:
                        features = self.faiss_index.index.reconstruct_n(0, self.faiss_index.ntotal)
                        dist_mat, matching = self.faiss_index.search(features,
                                                                     self.faiss_index.ntotal)
                        for i in range(self.faiss_index.ntotal):
                            dist_mat[i, :] = dist_mat[i, matching[i].argsort()]
                        self.distance_matrix = dist_mat
                        self.distance_matrix_indices = faiss.vector_to_array(
                            self.faiss_index.id_map)
                    else:
                        # Only update the elements that actually change
                        fill_up_index = np.argwhere(self.distance_matrix_indices < 0)[0, 0]
                        a, b = self.faiss_index.search(image_features, self.faiss_index.ntotal)
                        self.distance_matrix_indices[fill_up_index] = index
                        sorter = np.argsort(b[0])
                        sorter_idx = sorter[
                            np.searchsorted(b[0], self.distance_matrix_indices, sorter=sorter)]
                        a = a[:, sorter_idx][0]
                        self.distance_matrix[fill_up_index, :] = self.distance_matrix[:,
                                                                 fill_up_index] = a

                    # Subtract self-similarity
                    remove_index_tmp = np.argmax(
                        self.distance_matrix.sum(0) - self.distance_matrix.diagonal())
                    self.distance_matrix[:, remove_index_tmp] = self.distance_matrix[
                                                                remove_index_tmp,
                                                                :] = -1
                    remove_index = self.distance_matrix_indices[remove_index_tmp]
                    self.distance_matrix_indices[remove_index_tmp] = -1
                    self.faiss_index.remove_ids(np.array([remove_index]))
                    remove_sample = remove_index
                    if verbose:
                        print(f'Removed sample {remove_index} from the replay buffer')

        else:
            self.faiss_index.add_with_ids(image_features, np.array([index]))
            add_sample = True
            if self.faiss_index.ntotal > self.buffer_size:
                remove_index = self.target_sampler.choice(self.faiss_index.ntotal, 1)[0]
                remove_sample = faiss.vector_to_array(self.faiss_index.id_map)[remove_index]
                self.faiss_index.remove_ids(np.array([remove_sample]))
                # if verbose:
                #     print(f'Removed sample {remove_sample} from the target buffer')

        if add_sample:
            filename = self.storage_dir / f'{self.dataset_type}_{index:>05}.pkl'
            data = {
                key: value
                for key, value in sample.items() if 'index' in key or 'camera_matrix' in key
                                                    or 'inv_camera_matrix' in key
                                                    or 'relative_distance' in key
            }
            data['rgb', -1] = sample_filenames['images'][0]
            data['rgb', 0] = sample_filenames['images'][1]
            data['rgb', 1] = sample_filenames['images'][2]
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            self.online_filenames.append(filename)

        if remove_sample is not None:
            for filename in self.online_filenames:
                if f'_{remove_sample:>05}.pkl' in filename.name:
                    os.remove(filename)
                    self.online_filenames.remove(filename)
                    break

    def get(self, sample: Dict[str, Any], image_features: Optional[Tensor] = None) -> Dict[
        str, Any]:
        return_data = {}

        # Sample from target buffer
        if self.online_filenames and self.batch_size > 0:
            index = sample['index'].item() + self.faiss_index_offset
            filename = self.storage_dir / f'{self.dataset_type}_{index:>05}.pkl'
            # The current sample is the only one that is in the buffer
            if len(self.online_filenames) == 1 and filename in self.online_filenames:
                replace = True
                num_samples = 1
                sampling_prob = None
            else:
                # Do not sample the current sample
                if filename in self.online_filenames:
                    num_samples = len(self.online_filenames) - 1  # -1 for the current sample
                else:
                    num_samples = len(self.online_filenames)
                replace = self.batch_size > num_samples

                if self.similarity_sampling:
                    assert self.faiss_index.ntotal > 0
                    if image_features is None:
                        image_features = self.feature_encoder(
                            sample['rgb', 0, 0]).detach().cpu().numpy()
                    faiss.normalize_L2(
                        image_features)  # The inner product becomes cosine similarity
                    similarity, indices = self.faiss_index.search(image_features,
                                                                  self.faiss_index.ntotal)
                    if index in indices:
                        similarity = np.delete(similarity, np.argwhere(indices == index))
                    else:
                        similarity = similarity[0]
                    dissimilarity = 1 - similarity
                    # sampling_prob = dissimilarity / dissimilarity.sum()
                    sampling_prob = similarity / similarity.sum()
                else:
                    sampling_prob = None

            indices = self.target_sampler.choice(num_samples, self.batch_size, replace,
                                                 sampling_prob)
            filenames = [self.online_filenames[index] for index in indices]
            return_data = self._get(filenames[0])
            for filename in filenames[1:]:
                data = self._get(filename)
                for key in return_data:
                    return_data[key] = torch.cat([return_data[key], data[key]])

        return return_data

    def save_state(self):
        filename = self.storage_dir / 'buffer_state.pkl'
        data = {'filenames': self.online_filenames, 'faiss_index': self.faiss_index}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Saved reply buffer state to: {filename}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

    def load_state(self, state_path: Path):
        with open(state_path, 'rb') as f:
            data = pickle.load(f)
            # self.buffer_filenames = data['filenames']
            self.faiss_index = data['faiss_index']
            self.faiss_index_offset = faiss.vector_to_array(self.faiss_index.id_map).max()
            self.online_filenames = [state_path.parent / file.name for file in data['filenames']]
        print(f'Load replay buffer state from: {state_path}')
        for key, value in self.buffer_filenames.items():
            print(f'{key + ":":<12} {len(value):>5}')

    def __getitem__(self, index: int) -> Dict[Any, Tensor]:
        raise NotImplementedError

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

    def _reset_storage_dir(self):
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
