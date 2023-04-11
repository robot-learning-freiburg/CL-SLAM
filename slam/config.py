import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Slam:
    config_file: Path
    dataset_sequence: int
    adaptation: bool
    adaptation_epochs: int
    min_distance: float
    start_frame: int
    logging: bool
    do_loop_closures: bool
    keyframe_frequency: int
    lc_distance_poses: int

@dataclasses.dataclass
class ReplayBuffer:
    config_file: Path
    maximize_diversity: bool
    max_buffer_size: int
    similarity_threshold: float
    similarity_sampling: bool
    load_path: Path
