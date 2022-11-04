import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from .melon import MelonGenreDataset

def get_dataloaders(args):
    melon_genre_set = MelonGenreDataset(args['path_melon'], args['melon_kfold_ver'])

    train_set = TrainSet(melon_genre_set.train_set, args['crop_size'])
    val_set = TestSet(melon_genre_set.val_set, args['num_seg'], args['crop_size'])
    eval_set = TestSet(melon_genre_set.eval_set, args['num_seg'], args['crop_size'])

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

    return train_loader, val_loader, eval_loader, melon_genre_set.genres

class TrainSet(Dataset):
    def __init__(self, items, crop_size):
        self.items = items
        self.crop_size = crop_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        data = np.load(item.path)

        length = data.shape[-1]
        if length < self.crop_size:
            shortage = self.crop_size - length
            data = np.pad(data, ((0, 0), (0, shortage)), 'wrap')
        else:
            index = random.randint(0, length - self.crop_size)
            data = data[:, index:index + self.crop_size]

        return data, item.genre

class TestSet(Dataset):
    def __init__(self, items, num_seg, crop_size):
        self.items = items
        self.crop_size = crop_size
        self.num_seg = num_seg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        data = np.load(item.path)

        # crop
        length = data.shape[-1]
        if length < self.crop_size:
            shortage = self.crop_size - length
            data = np.pad(data, ((0, 0), (0, shortage)), 'wrap')

        # stack
        buffer = []
        indices = np.linspace(0, data.shape[-1] - self.crop_size, self.num_seg)
        for idx in indices:
            idx = int(idx)
            buffer.append(data[:, idx:idx + self.crop_size])
        buffer = np.stack(buffer, axis=0)
        
        return buffer, item.genre