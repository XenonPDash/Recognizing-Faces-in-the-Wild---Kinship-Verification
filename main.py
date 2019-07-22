
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

import net_sphere
from cross_entropy_loss import CrossEntropyLoss
from data_loader import get_train_loader, get_val_loader
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


def validate(net, net2, loader):
    print('Begin validation')
    val_loss = 0
    net.eval()
    net2.eval()

    # csv_path = None

    # loader = get_val_loader(base_dir, csv_path)

    dists = []
    final_correct = 0
    final_count = 0
    for batchid, (inputs, targets) in enumerate(loader):

        input0 = inputs[0]
        input1 = inputs[1]
        if use_gpu: input0, input1, targets = input0.cuda(), input1.cuda(), targets.cuda()
        input0, input1, targets = Variable(input0), Variable(input1), Variable(targets)
        # img_b = Tensor(pairs[1])
        # _, embs_a = net(img_a)
        # _, embs_b = net(img_b)

        embs_a = net(input0)
        embs_b = net(input1)

        output = net2(embs_a, embs_b)

        mask = output.ge(0.5).float()
        mask = mask.squeeze(1)
        correct = (mask == targets.float()).sum()
        final_correct = final_correct + correct.item()
        final_count = final_count + targets.size(0)

        targets = targets.float()
        loss = criterion3(output, targets)

        lossd = loss.item()
        del loss
        val_loss += lossd


    accuracy = final_correct / final_count
    print('ACC={:.4f}'.format(accuracy))

    val_loss = val_loss / (batchid + 1)
    print('val loss={:.4f}'.format(val_loss))


    return accuracy, val_loss


def train(net, net2, optimizer, epoch, loader):
    net.train()
    net2.train()
    train_loss = 0
    correct = 0
    total = 0
    for batchid, (inputs, targets) in enumerate(loader):
        input0 = inputs[0]
        input1 = inputs[1]
        if use_gpu: input0, input1, targets = input0.cuda(), input1.cuda(), targets.cuda()

        optimizer.zero_grad()
        input0, input1, targets = Variable(input0), Variable(input1), Variable(targets)
        # outputs, _ = net(inputs)
        output0 = net(input0)
        output1 = net(input1)

        # loss = criterion(outputs, targets)

        # loss, _ = criterion3(outputs, targets)
        # loss = criterion(outputs, pids)

        output = net2(output0, output1)
        output = output.squeeze(1)
        # outputs = torch.cat((output0, output1), dim=0)
        targets = targets.float()
        loss = criterion3(output, targets)

        loss.backward()
        optimizer.step()
        lossd = loss.item()
        del loss
        train_loss += lossd
        # _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()


        if batchid % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batchid, len(loader),
                                                              train_loss / (batchid + 1), lossd))

    print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FIW Sphereface Baseline')
    parser.add_argument('--net', '-n', default='sphere20a', type=str)

    parser.add_argument('--train', action='store_true', help='set to not train')
    parser.add_argument('--finetune', action='store_true', help='set to fine-tune the pretrained model')
    parser.add_argument('--pretrained', default='model/sphere20a_20171020.pth', type=str,
                        help='the pretrained model to point to')
    parser.add_argument('--data_dir', '-d', type=str, default='',
                        help='Root directory of data (assumed to contain traindata and valdata)')
    parser.add_argument('--classnum', type=int, default='2232', help='number of calss')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--use-cpu', action='store_true', help='use cpu')
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--print-freq', type=int, default=30, help="print frequency")
    # ======================================================================
    ### 参数调整区
    # 显卡
    parser.add_argument('--gpu-devices', type=str, default='1', help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # 图片大小
    parser.add_argument('--img_size', default=(224, 224), type=tuple, help='input image size')
    # train batch size
    parser.add_argument('--train_batch_size', default=16, type=int, help='training batch size')
    # val batch size
    parser.add_argument('--val_batch_size', default=16, type=int, help='val batch size')
    # train step
    parser.add_argument('--train_steps', default=200, type=int, help='training steps')
    # val step
    parser.add_argument('--val_steps', default=100, type=int, help='val steps')
    # 初始lr
    parser.add_argument('--lr', default=0.00001, type=float, help='inital learning rate')
    # epoch数
    parser.add_argument('--n_epochs', default=200, type=int, help='number of training epochs')
    # 在第几个epoch更改lr
    parser.add_argument('--change_lr_for_epochs', default=[50, 100, 150], type=list, help='epoch number to change lr')
    # training set中1:0的比例
    parser.add_argument('--one_to_zero_train', default=1 / 1, type=float, help='1 to 0 label ratio in training set')
    # val set中1:0的比例
    parser.add_argument('--one_to_zero_val', default=1 / 1, type=float, help='1 to 0 label ratio in val set')
    # ======================================================================

    # ======================================================================
    # reid models args
#    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # ======================================================================

    args = parser.parse_args()

    # print(args.data_dir)

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

    # Create model
    #    net = getattr(net_sphere, args.net)(classnum=args.classnum)

#    net = models.create(args.arch, num_features=1024, dropout=args.dropout, num_classes=args.features)

    net = ResNet.resnet50(num_classes=8631, include_top=False)
    weights_path = 'weight_file/resnet50_ft_weight.pkl'    
    with open(weights_path, 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    net.load_state_dict(weights)


    # net.classifier[6] = nn.Linear(4096, args.classnum)
    #    for name, param in net.named_parameters():
    #        param.requires_grad = False
    # net.fc = nn.Linear(2048, 1024)

    net2 = net_classification()

    # model_state = net.state_dict()



    if use_gpu:
        net = nn.DataParallel(net).cuda()
        net2 = net2.cuda()

    # criterion = net_sphere.AngleLoss()
    # criterion = nn.Softmax()

    # criterion = CrossEntropyLoss(num_classes=args.classnum, use_gpu=use_gpu, label_smooth=True)

    # train_dir = '/Users/josephrobinson/Downloads/'
    train_dir = args.data_dir
    # print(train_dir)
    val_dir = args.data_dir
    # 'train'

    # criterion2 = TripletLoss(margin=0.3, train_set=train_set)
    criterion3 = nn.BCELoss().cuda()
    params = [x for x in net.parameters() if x.requires_grad]
    params2 = [x for x in net2.parameters() if x.requires_grad]
    param_groups = [
        {'params': params, 'lr_mult': 1.0},
        {'params': params2, 'lr_mult': 1.0}]

#    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience=20, verbose=True)
    print('start: time={}'.format(dt()))

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_acc = 0
    if not args.train:
        print('Begin train')
        for epoch in range(args.n_epochs):
            train_set, train_loader = get_train_loader(image_size=args.img_size,
                                                       batch_size=args.train_batch_size,
                                                       train_steps=args.train_steps,
                                                       val_steps=args.val_steps,
                                                       one_to_zero_train=args.one_to_zero_train,
                                                       one_to_zero_val=args.one_to_zero_val)
            val_loader = get_val_loader(image_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        train_steps=args.train_steps,
                                        val_steps=args.val_steps,
                                        one_to_zero_train=args.one_to_zero_train,
                                        one_to_zero_val=args.one_to_zero_val)
            print("epoch:", epoch)

        #    if epoch in args.change_lr_for_epochs:
        #        args.lr *= 0.1
        #        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4)

            train(net, net2, optimizer, epoch, train_loader)
            acc,val_loss = validate(net, net2, val_loader)
            print('accuracy = ', acc)
            scheduler.step(val_loss)

            if best_acc < acc:
                best_acc = acc
                # =====================================================
                # save check point
                if use_gpu:
                    state_dict1 = net.state_dict()
                    state_dict2 = net2.state_dict()
                else:
                    state_dict1 = net.state_dict()
                    state_dict2 = net2.state_dict()

                state1 = {'net': state_dict1,
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch + 1}
                state2 = {'net': state_dict2,
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch + 1}
                checkpoint_dir1 = 'check_point/checkpoint_net1.pth'
                checkpoint_dir2 = 'check_point/checkpoint_net2.pth'
                # mkdir_if_missing(osp.dirname('check_point'))
                torch.save(state1, checkpoint_dir1)
                torch.save(state2, checkpoint_dir2)
                # =====================================================

    print('finish: time={}\n'.format(dt()))

