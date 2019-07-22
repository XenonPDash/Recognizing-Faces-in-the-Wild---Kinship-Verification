from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import csv
from tools import _pluck, findStr, val_label, json_to_dict, img_paths_to_kinship
from sampler import RandomIdentitySampler
from collections import defaultdict
from glob import glob
import pandas as pd
from random import choice, sample


class FIW_DBBase(Dataset):
    def __init__(self, n_classes=300, transform=None):
        # self.root_dir = root_dir
        # self.labels_path = labels_path
        self.n_classes = n_classes
        self.transform = transform
        self.pairs = []
        self.__gettestset__()
        self.csv_dict = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pass

    def __gettestset__(self):
        submission = pd.read_csv('input/sample_submission.csv')
        test_set = []
        for pairs in submission.img_pair.values:
            path_suffix = "input/test/"
            test_tuple = (path_suffix + str(pairs.split("-")[0]), path_suffix + str(pairs.split("-")[1]))
            test_set.append(test_tuple)
        self.pairs = test_set


class FIW_Test(FIW_DBBase):
    """Dataset class for FIW Training Set"""

    def preprocess(self):
        """Process the labels file"""
        pass

    def __getitem__(self, indices):
        """Return an image"""
        img_tuple = self.pairs[indices]
        img_path1 = img_tuple[0]
        img_path2 = img_tuple[1]
        img_1 = self.transform(Image.open(img_path1))
        img_2 = self.transform(Image.open(img_path2))
        return (img_1, img_2)




def get_test_loader(n_classes=300, image_size=(224, 224), batch_size=16,
                     num_workers=4, use_gpu=True):
    """Build and return a data loader for the training set."""
    transform = T.Compose([T.RandomHorizontalFlip(),
                           T.Resize(image_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])
    dataset = FIW_Test(n_classes=n_classes, transform=transform)
    dataset.preprocess()

    pin_memory = True if use_gpu else False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=False)
    return data_loader


