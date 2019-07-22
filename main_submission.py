
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.backends import cudnn

import os
import sys
import os.path as osp
import datetime
import math
import argparse
import numpy as np
from torch import Tensor
import pandas as pd

import net_sphere
from cross_entropy_loss import CrossEntropyLoss
from data_loader_submission import get_test_loader
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from utils.logging import Logger
from utils.osutils import mkdir_if_missing
from triplet import TripletLoss
import models.resnet as ResNet
from net_classification import net_classification
import pickle

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r' + s)
    sys.stdout.flush()


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def KFold(n, n_folds=5, shuffle=False):
    folds = []
    if shuffle == False:
        base = list(range(n))
    else:
        base = torch.randperm(n)
        base = base.numpy().tolist()
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int(math.ceil((i + 1) * n / n_folds))]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = True if float(d[0]) > threshold else False
        y_predict.append(same)
        y_true.append(bool(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    tp = np.sum(np.logical_and(y_predict, y_true))
    fp = np.sum(np.logical_and(y_predict, np.logical_not(y_true)))
    tn = np.sum(np.logical_and(np.logical_not(y_predict), np.logical_not(y_true)))
    fn = np.sum(np.logical_and(np.logical_not(y_predict), y_true))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(y_true)

    return tpr, fpr, acc


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        _, _, accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def predict(net, net2, loader):
    print('Begin prediction')

    net.eval()
    net2.eval()

    predictions = []
    for pairno, inputs in enumerate(loader):
        print("pairno = ", pairno)
        input0 = inputs[0]
        input1 = inputs[1]
        if use_gpu: input0, input1 = input0.cuda(), input1.cuda()
        input0, input1 = Variable(input0), Variable(input1)

        embs_a = net(input0)
        embs_b = net(input1)

        output = net2(embs_a, embs_b)
        output = output.detach()
        mask = output.float()
        mask = mask.squeeze(1)
        predictions.extend(mask.cpu().numpy().tolist())
    print(predictions)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--net', '-n', default='sphere20a', type=str)
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')

    parser.add_argument('--train', action='store_true', help='set to not train')
    parser.add_argument('--pretrained', default='model/sphere20a_20171020.pth', type=str,
                        help='the pretrained model to point to')
    parser.add_argument('--data_dir', '-d', type=str, default='',
                        help='Root directory of data (assumed to contain traindata and valdata)')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--use-cpu', action='store_true', help='use cpu')
    parser.add_argument('--gpu-devices', type=str, default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--print-freq', type=int, default=30, help="print frequency")

    # ======================================================================
    # reid models args
    #parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # ======================================================================

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # 建立网络
    net = ResNet.resnet50(num_classes=8631, include_top=False)
    weights_path = 'weight_file/resnet50_ft_weight.pkl'    
    with open(weights_path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    net.load_state_dict(weights)
    net = nn.DataParallel(net)
    net2 = net_classification()
    # 读取weight
    checkpoint1 = torch.load("check_point/checkpoint_net1.pth")
    checkpoint2 = torch.load("check_point/checkpoint_net2.pth")



    state_dict =checkpoint1['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v

    
    net.load_state_dict(new_state_dict)
    net2.load_state_dict(checkpoint2['net'])

    if use_gpu:
        net = net.cuda()
        net2 = net2.cuda()

    print('start: time={}'.format(dt()))

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_acc = 0
    print('Begin test')
    test_loader = get_test_loader(batch_size=args.batch_size)

    predictions = predict(net, net2, test_loader)
    submission = pd.read_csv('input/sample_submission.csv')
    submission['is_related'] = predictions
    submission.to_csv("results.csv", index=False)

    print('finish: time={}\n'.format(dt()))

