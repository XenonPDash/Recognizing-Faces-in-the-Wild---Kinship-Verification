import numpy as np
import argparse
from sphereface_rfiw_baseline import net_sphere
import os
import torch
# from torchtools import cuda, TorchTools, Tensor
from sphereface_rfiw_baseline.data_loader import get_val_loader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import Tensor
# from fiwtools.utils.io import sys_home
import torch.nn.functional as F
from torchvision import models

do_plot = True


def initialize_roc_plot(ax, lw=2):
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # ax.xlabel('False Positive Rate')
    # ax.ylabel('True Positive Rate')
    # ax.title('Receiver operating characteristic example')
    # ax.legend(loc="lower right")
    return ax


def generate_roc(ax, tp_array, fp_array, roc_auc, color='darkred', lw=2, label='ROC curve (area = %0.2f)', init=False,
                 fname=''):
    # if init:
    #     initialize_roc_plot(lw)
    ax.plot(fp_array, tp_array, color=color, lw=lw, label=label % roc_auc)
    # if len(fname) > 0:
    #     plt.savefig(fname)


def validate(net, data_loader, ax):
    print('Begin validation')
    net.eval()
    y_labels = []
    distances = []
    ii = 0
    for pairs, labels in iter(data_loader):
        if ii > 10:
            break
        img_a = Variable(pairs[0]).type(Tensor)
        img_b = Variable(pairs[1]).type(Tensor)
        if cuda: img_a, img_b = img_a.cuda(), img_b.cuda()

        _, embs_a = net(img_a)
        _, embs_b = net(img_b)

        embs_a = embs_a.data
        embs_b = embs_b.data
        cos_dis = F.cosine_similarity(embs_a, embs_b)
        distances += list(cos_dis.data.cpu().numpy())

        y_labels += list(labels.numpy())
        # ii += 1

    dist_array = np.array(distances)
    y_array = np.array(y_labels)

    fpr, tpr, thresh = roc_curve(y_array, dist_array)
    roc_auc = auc(fpr, tpr)

    if do_plot:
        initialize_roc_plot(ax)
        generate_roc(ax, tpr, fpr, roc_auc, color='darkred')
        # fpr_micro, tpr_micro, _ = roc_curve(y_array.ravel(), dist_array.ravel())
        # roc_auc_micro = auc(fpr_micro, tpr_micro)
        #
        # generate_roc(tpr_micro, fpr_micro, roc_auc_micro, color='darkorange')

        # plt.show()

    return roc_auc


do_types = np.linspace(0, 6, 7).astype(np.uint8)
types = ['bb', 'ss', 'sibs', 'fd', 'fs', 'md', 'ms']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--type', '-t', default='bb', type=str,
                        help='relationship type (None processes entire directory)')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--modelpath', default='finetuned/checkpoint.pth.tar', type=str,
                        help='the pretrained model to point to')


    args = parser.parse_args()

    cuda = True

    # net = net_sphere.sphere20a(classnum=2232)
    net = models.vgg16()
    if cuda:
        net.cuda()

    # ====================================================
    # 读取网络参数
    # epoch, bess_acc = TorchTools.load_checkpoint(net, f_weights=args.modelpath)
    checkpoint_dir = 'check_point/checkpoint.pth'
    checkpoint = torch.load(checkpoint_dir)
    net.load_state_dict(checkpoint['net'])
    # ====================================================



    # ncols = int(np.ceil(len(do_types) / 2))
    ncols = 1
    nrows = 1

    f, axes = plt.subplots(nrows, ncols, sharex='all', sharey='all')

    ax = axes
    loader = get_val_loader(batch_size=args.batch_size)
    # f.subplot()
    auc_score = validate(net, loader, ax)

    print('{} (auc)'.format(auc_score))

    plt.savefig('roc.png')
