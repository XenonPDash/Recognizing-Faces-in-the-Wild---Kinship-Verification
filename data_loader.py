
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
    def __init__(self,
                 n_classes=300,
                 train_steps=200,
                 val_steps=100,
                 one_to_zero_train=1 / 1,
                 one_to_zero_val=1 / 1,
                 transform=None):
        # self.root_dir = root_dir
        # self.labels_path = labels_path
        self.n_classes = n_classes
        self.transform = transform
        self.pairs = []
        self.val_pairs = []
        self.tr_pairs = []
        self.__getallset__(train_steps=train_steps,
                           val_steps=val_steps,
                           train_batch_size=16,
                           val_batch_size=16,
                           one_to_zero_train=one_to_zero_train,
                           one_to_zero_val=one_to_zero_val)
        self.csv_dict = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pass

    def __getallset__(self, train_steps, val_steps, train_batch_size, val_batch_size, one_to_zero_train,
                      one_to_zero_val):
        # training set label的路径
        train_file_path = "input/train_relationships.csv"
        # training set 文件路径
        train_folders_path = "input/train/"
        # 抽取F0900 - F0999家庭作为val set
        val_famillies = "F09"

        # 读取所有图片列表并划分training set和test set
        all_images = glob(train_folders_path + "*/*/*.jpg")
        all_images = [x.replace("\\", "/") for x in all_images]
        train_images = [x.replace("\\", "/") for x in all_images if val_famillies not in x]
        val_images = [x.replace("\\", "/") for x in all_images if val_famillies in x]

        # 获得所有图片中的人名部分（如F0002/MID1）有重复
        ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

        # 将training set中的所有图片地址按人名划分为dict
        train_person_to_images_map = defaultdict(list)
        for x in train_images:
            train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

        # 将val set中的所有图片地址按人名划分为dict
        val_person_to_images_map = defaultdict(list)
        for x in val_images:
            val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

        # 获取所有已知的亲属关系（人名对的形式），并剔除没有对应文件存在的人名对
        relationships = pd.read_csv(train_file_path)
        relationships = list(zip(relationships.p1.values, relationships.p2.values))
        relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]  # 过滤csv，使其只包含存在文件的人名

        # 划分relationship人名对，分为training set组和validation set组
        train = [x for x in relationships if val_famillies not in x[0]]
        val = [x for x in relationships if val_famillies in x[0]]

        train_set = []
        val_set = []
        for i in range(train_steps):
            train_set.extend(
                self.__gen(list_tuples=train,
                           person_to_images_map=train_person_to_images_map,
                           batch_size=train_batch_size,
                           one_to_zero=one_to_zero_train))
        for i in range(val_steps):
            val_set.extend(
                self.__gen(list_tuples=val,
                           person_to_images_map=val_person_to_images_map,
                           batch_size=val_batch_size,
                           one_to_zero=one_to_zero_val))

        self.tr_pairs = train_set
        self.val_pairs = val_set

    def __gen(self, list_tuples, person_to_images_map, batch_size, one_to_zero=1):
        # 从传入的dict里面取出所有的人名（无重复）
        ppl = list(person_to_images_map.keys())
        # 从所有train set relationship中抽取半个batch的量的人名对，label为1
        batch_tuples = sample(list_tuples, int(batch_size * (one_to_zero / (one_to_zero + 1))))
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            # 从所有train set的任命钟随即取出两个，并要求这二人没有关系，label为0
            # 总共取出label为0的半个batch的量
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        # for x in batch_tuples:
        #     if not len(person_to_images_map[x[0]]):
        #         print(x[0])

        # 根据people_image_dict对每个人随机取图
        image_path_tuples = [(choice(person_to_images_map[x[0]]), choice(person_to_images_map[x[1]])) for x in
                             batch_tuples]
        grid = [image_path_tuples, labels]
        grid = [[row[i] for row in grid] for i in range(len(grid[0]))]

        return grid


class FIW_Train(FIW_DBBase):
    """Dataset class for FIW Training Set"""

    def preprocess(self):
        """Process the labels file"""
        self.pairs = self.tr_pairs

    def __getitem__(self, indices):
        """Return an image"""
        img_tuple, label = self.pairs[indices]
        img_path1 = img_tuple[0]
        img_path2 = img_tuple[1]
        img_1 = self.transform(Image.open(img_path1))
        img_2 = self.transform(Image.open(img_path2))
        return (img_1, img_2), label

    def _get_single_item(self, index):
        filename, label = self.pairs[index]
        impath = filename
        image = Image.open(impath)
        return self.transform(image), label


class FIW_Val(FIW_DBBase):
    """Dataset class for FIW Validation Set"""

    def preprocess(self):
        """Process the pair CSVs"""
        self.pairs = self.val_pairs

    def __getitem__(self, indices):
        """Return a pair"""
        img_tuple, label = self.pairs[indices]
        img_path1 = img_tuple[0]
        img_path2 = img_tuple[1]
        img_1 = self.transform(Image.open(img_path1))
        img_2 = self.transform(Image.open(img_path2))
        return (img_1, img_2), label

    def __len__(self):
        """Return the number of images."""
        return len(self.pairs)


def get_train_loader(image_size,
                     batch_size,
                     train_steps,
                     val_steps,
                     one_to_zero_train,
                     one_to_zero_val,
                     num_workers=4,
                     use_gpu=True):
    """Build and return a data loader for the training set."""
    transform = T.Compose([T.RandomHorizontalFlip(),
                           T.Resize(image_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                           ])
    dataset = FIW_Train(train_steps=train_steps,
                        val_steps=val_steps,
                        one_to_zero_train=one_to_zero_train,
                        one_to_zero_val=one_to_zero_val,
                        transform=transform)
    dataset.preprocess()

    pin_memory = True if use_gpu else False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=True)
    return dataset, data_loader


def get_val_loader(image_size,
                   batch_size,
                   train_steps,
                   val_steps,
                   one_to_zero_train,
                   one_to_zero_val,
                   num_workers=4,
                   use_gpu=True):
    """Build and return a data loader for a split in the validation set."""
    transform = T.Compose([T.Resize(image_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                           ])

    dataset = FIW_Val(train_steps=train_steps,
                      val_steps=val_steps,
                      one_to_zero_train=one_to_zero_train,
                      one_to_zero_val=one_to_zero_val,
                      transform=transform)
    dataset.preprocess()
    pin_memory = True if use_gpu else False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=False)
    return data_loader

