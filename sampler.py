from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler, RandomSampler
from people_picker import get_apnn_from_id_with_dict


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, all_pn_dict):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))
        """
        data_source: training set
        pid: 人的label
        index: 图在training set中的编号（不是地址）
        pids: 所有人的编号
        """
        # ======================================
        # 使用生成apnn方法时，取消注释第一句，但注释第二句, 同时更改传入参数apnn_dict为all_pn_dict
        self.all_pn_dict = all_pn_dict
        # self.apnn_dict = apnn_dict
        # ======================================

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = self.pids.__len__() * self.num_instances * self.num_instances * 2

    def __iter__(self):

        """
        ===========================================================================================
        pids: 所有人的id
        idxs：属于某个pid的人的所有图的图id
        这一段要把每个人的所有图四四分组
        本身不足四张图的人，有放回地抽取四张，本身超过四张但不是4的倍数张的人，舍弃余数
        分组结果存入dict，为batch_idxs_dict
        结构为
        batch_idxs_dict = {
            人id1:[4个图id][4个图id][4个图id]
            人id1:[4个图id][4个图id]
            ...
        }
        """
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        """"
        ===========================================================================================
        """

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        selected_pids = []
        # print("=" * 70)
        # print(avai_pids.__len__())
        for avai_pid in avai_pids:
            # ==========================================================================
            # 使用生成apnn方法时，取消注释第1,2句，但注释第3句
            apnn = get_apnn_from_id_with_dict(avai_pid, self.data_source, self.all_pn_dict)
#            print("avai_pid = {}, apnn = {}".format(avai_pid, apnn))
            # apnn = self.apnn_dict[str(avai_pid)]
            # ==========================================================================

            a_pic_indices = random.choice(batch_idxs_dict[apnn[0]])
            p_pic_indices = random.choice(batch_idxs_dict[apnn[1]])
            n_pic_indices = random.choice(batch_idxs_dict[apnn[2]])
            n_pic_indices2 = random.choice(batch_idxs_dict[apnn[3]])
            n_pic_indices3 = random.choice(batch_idxs_dict[apnn[4]])


            ap_pairs = []
            an_pairs = []

            for a_index in a_pic_indices:
                for p_index in p_pic_indices:
                    ap_pairs.append([a_index, p_index])
            for a_index in a_pic_indices:
                for n_index in n_pic_indices:
                    an_pairs.append([a_index, n_index])


            final_idxs.extend(ap_pairs)
            final_idxs.extend(an_pairs)
        print(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
