import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from .melon import MelonGenreDataset

def get_dataloaders(args):
    melon_genre_set = MelonGenreDataset(args['path_melon'], args['melon_kfold_ver'])

    train_set = TrainSet(melon_genre_set.train_set, args['crop_size'])
    val_set = TestSet(melon_genre_set.val_set)
    eval_set = TestSet(melon_genre_set.eval_set)

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
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        data = np.load(item.path)

        return data, item.genre