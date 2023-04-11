import random

import numpy as np
import torch
from tqdm import tqdm

from config.config_parser import ConfigParser
from slam import Slam
from slam.utils import calc_error

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ============================================================
config = ConfigParser('./config/config_adapt.yaml')
print(config)

# ============================================================
slam = Slam(config)

with tqdm(desc='SLAM', total=len(slam)) as pbar:
    while slam.current_step < len(slam):
        losses = slam.step()
        pbar.set_postfix(depth=f'{losses["depth_loss"]:.5f}', velo=f'{losses["velocity_loss"]:.5f}')
        pbar.update(1)

if slam.do_adaptation:
    slam.save_model()
if slam.logging:
    slam.plot_metrics()
    slam.plot_trajectory()
    slam.pose_graph.visualize_in_meshlab(slam.log_path / 'pose_graph.obj', verbose=True)
    slam.gt_pose_graph.visualize_in_meshlab(slam.log_path / 'gt_pose_graph.obj', verbose=True)
error_log = calc_error(slam.pose_graph.get_all_poses(), slam.gt_pose_graph.get_all_poses())
print(error_log)

with open(config.depth_pose.log_path / 'log.txt', 'a', encoding='utf-8') as file:
    file.write(error_log)
