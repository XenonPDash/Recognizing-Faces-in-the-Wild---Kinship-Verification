from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from collections import defaultdict
from tools import people_ids_to_kinship
import pdb

class TripletLoss(nn.Module):
    def __init__(self, train_set, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.train_set = train_set

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
#        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # batch_num_ids = targets.size(0)/4
        pos_result_dic = defaultdict(list)
        neg_result_dic = defaultdict(list)
        temp_batch_idlist = targets.tolist()
        batch_idlist = []
        for i in range(0, len(temp_batch_idlist), 4):
            batch_idlist.append(temp_batch_idlist[i])
       # batch_idlist.sort(key=targets.tolist().index)
        for a_index, a_id in enumerate(batch_idlist):
            for q_index, q_id in enumerate(batch_idlist):
                #do the verification
                kinship = people_ids_to_kinship(a_id, q_id, self.train_set.pairs)
                if kinship == 1:
                    pos_result_dic[a_id].append(q_id)
                elif kinship == 0:
                    neg_result_dic[a_id].append(q_id)

        result_ten = torch.Tensor()
       # for a_index in range(0, len(batch_idlist), 4):
       #     a_id = batch_idlist[a_index]
        for a_index, a_id in enumerate(batch_idlist):
            row_list = []
            for q_index, q_id in enumerate(batch_idlist):
                if q_id in neg_result_dic[a_id]:
                    q_tensor = torch.zeros(4, 4)
                elif q_id in pos_result_dic[a_id]:
                    q_tensor = torch.ones(4, 4)
                else:
                    q_tensor = -torch.ones(4, 4)
                row_list.append(q_tensor)

            row_tensor = torch.cat(row_list, dim=1)

            result_ten = torch.cat((result_ten, row_tensor), dim=0)
        mask = result_ten
#        pdb.set_trace()
        dist_ap, dist_an = [], []
        for k in range(n):
            if k%16 == 0:
                for i in range(k, k+4):
#                    pdb.set_trace() 
                    dist_ap.append(dist[i][mask[i] == 1].max().unsqueeze(0))
                    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    #    print(dist_ap)
#        pdb.set_trace()
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
