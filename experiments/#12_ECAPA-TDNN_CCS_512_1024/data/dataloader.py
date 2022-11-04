import soundfile
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .GTZAN import GTZAN

def get_dataloaders(args):
    gtzan = GTZAN(args['path_gtzan'], args['kfold_ver'])

    train_set = TrainSet(gtzan.train_list, args['crop_size'], args['data_cycle'])
    val_set = TestSet(gtzan.val_list)
    eval_set = TestSet(gtzan.eval_list)

    train_loader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        pin_memory=True,
        num_workers=args['num_workers'],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        pin_memory=True,
        num_workers=args['num_workers'],
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        pin_memory=True,
        num_workers=args['num_workers'],
    )

    return train_loader, val_loader, eval_loader, gtzan.genres

class TrainSet(Dataset):
    def __init__(self, train_list, crop_size, cycle):
        self.items = train_list
        self.crop_size = int(crop_size * 220.5 + 256)
        self.cycle = cycle
        self.data_len = len(train_list)

    def __len__(self):
        return self.data_len * self.cycle

    def __getitem__(self, index):
        index = index % self.data_len
        item = self.items[index]

        data,_ = soundfile.read(item.path)
        
        length = data.shape[-1]
        if length < self.crop_size:
            shortage = self.crop_size - length
            data = np.pad(data, ((0, 0), (0, shortage)), 'wrap')
        else:
            index = random.randint(0, length - self.crop_size)
            data = data[index:index + self.crop_size]

        return torch.FloatTensor(data), item.genre

class TestSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        data,_ = soundfile.read(item.path)

        return torch.FloatTensor(data), item.genre