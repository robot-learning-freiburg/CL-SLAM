from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Cityscapes
from slam.replay_buffer import ReplayBuffer

# ============================================================

replay_buffer_path = Path(
    __file__).parent / 'log/cityscapes/replay_buffer'  # <-- MATCH WITH config_pretrain.yaml
replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
replay_buffer = ReplayBuffer(replay_buffer_path, 'Cityscapes')

# ============================================================

dataset = Cityscapes(
    Path('USER/data/cityscapes'),  # <-- ADJUST THIS
    'train',
    [-1, 0, 1],
    [0, 1, 2, 3],
    192,
    640,
    do_augmentation=False,
    views=('left', ),
)
dataloader = DataLoader(dataset, num_workers=12, batch_size=1, shuffle=False, drop_last=True)

# ============================================================

with tqdm(total=len(dataloader)) as pbar:
    for i, sample in enumerate(dataloader):
        replay_buffer.add(sample, dataset.get_item_filenames(i))
        pbar.update(1)
replay_buffer.save_state()
