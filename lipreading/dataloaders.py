import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate
#15:43

def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
        Normalize(0.0, 255.0),
        RandomCrop(crop_size),
        HorizontalFlip(0.5),
        Normalize(mean, std),
        TimeMask(T=0.6*25, n_mask=1)
    ])

    preprocessing['val'] = Compose([
        Normalize(0.0, 255.0),
        CenterCrop(crop_size),
        Normalize(mean, std)])

    preprocessing['test'] = preprocessing['val']

    return preprocessing


def get_data_loaders(data_dir, label_path, batch_size, num_workers, test_only=False, annonation_direc=None):
    preprocessing = get_preprocessing_pipelines()

    # create dataset object for each partition
    partitions = ['test'] if test_only else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
        data_partition=partition,
        data_dir=data_dir,
        label_fp=label_path,
        annonation_direc=annonation_direc,
        preprocessing_func=preprocessing[partition],
        data_suffix='.npz',
        use_boundary=True,
    ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
