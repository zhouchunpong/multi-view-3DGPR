import argparse
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
import time
import os
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from data2 import get_problem, subset_random_sampler
from models.densenet_SwitOKD_Attention_base import densenet121
import csv
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', type=str, nargs='+', default=['DenseNet121', 'DenseNet121'])
parser.add_argument('--type', type=str, default='SwitOKD')
parser.add_argument('--version', type=str, default='Attention_fusion_plus')

parser.add_argument('--data_dir', type=str, default='./radar_data')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--epoch', type=int, default=360)
parser.add_argument('--ratio', type=int, default=[0.6, 0.2, 0.2])

parser.add_argument('--DA', type=bool, default=True)
parser.add_argument('--aug_times', type=int, default=0)

parser.add_argument('--T', type=float, default=4.0)  # temperature
parser.add_argument('--w_top', type=float, default=0.2)  # weight for ce and kl
parser.add_argument('--w_main', type=float, default=0.2)  # weight for ce and kl

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_top', type=float, default=1e-3)
parser.add_argument('--lr_main', type=float, default=1e-3)
parser.add_argument('--lr_other', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.2)

parser.add_argument('--milestones', type=int, nargs='+', default=[])
parser.add_argument('--milestones_top', type=int, nargs='+', default=[280])
parser.add_argument('--milestones_main', type=int, nargs='+', default=[280])
parser.add_argument('--milestones_other', type=int, nargs='+', default=[280])

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu-id', type=int, default=4)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--pic_name', type=str, default='')
parser.add_argument('--pic_name1', type=str, default='top')
parser.add_argument('--pic_name2', type=str, default='main')
parser.add_argument('--pic_name3', type=str, default='fusion')

parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')

args = parser.parse_args()
args.num_branch = len(args.model_names)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# torch.cuda.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


exp_name = '_'.join(args.model_names) + '_' + args.type + '_' + args.version
exp_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
exp_path = './experiments/{}/{}'.format(exp_name, exp_time)
os.makedirs(exp_path, exist_ok=True)
sys.stdout = Logger(exp_path + '/' + exp_name + '.txt')
print(exp_path)


def draw_acc_loss(Loss, Val_Loss, Accuracy, Val_Accuracy, pic_name):
    plt.figure()
    plt.plot(Loss, label="Train loss")
    plt.plot(Val_Loss, label="Val loss")
    plt.title("Average loss vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. train. loss")
    plt.legend()
    plt.savefig(exp_path + '/loss_' + exp_name + '_' + pic_name + '.png')

    plt.figure()
    plt.plot(Accuracy, label="Train accuracy")
    plt.plot(Val_Accuracy, label="Val accuracy")
    plt.title("Average accuracy vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. accuracy")
    plt.legend()
    plt.savefig(exp_path + '/acc_' + exp_name + '_' + pic_name + '.png')


def draw_test_acc_loss(Test_Loss, Test_Accuracy, pic_name):
    plt.figure()
    plt.plot(Test_Accuracy, label="Test accuracy")
    plt.title("Average accuracy vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. accuracy")
    plt.legend()
    plt.savefig(exp_path + '/acc_' + exp_name + '_' + pic_name + '.png')
    for i in range(Test_Accuracy):
        writer.add_scalar(
            tag="acc" + pic_name,  # 可以暂时理解为图像的名字
            scalar_value=Test_Accuracy[i],  # 纵坐标的值
            global_step=i  # 当前是第几次迭代，可以理解为横坐标的值
        )


def kl_div(p_logit, q_logit):
    p = F.softmax(p_logit / args.T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / args.T, dim=-1)
                        - F.log_softmax(q_logit / args.T, dim=-1)), 1)
    return torch.mean(kl)


def dist_s_label(y, q):
    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)
    return torch.mean(dist)


def dist_s_t(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, args.consistency_rampup)


def train(model, device, epochs, data_loaders, type):
    model.to(device)
    # 定义loss function和优化器
    all_params = model.parameters()
    dn1_params = model.dn1.parameters()  # list(map(id,model.dn1.parameters()))
    dn2_params = model.dn2.parameters()
    # 取回分组参数的id
    params_id = list(map(id, dn1_params)) + list(map(id, dn2_params))
    # 取回剩余分特殊处置参数的id
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # 构建不同学习参数的优化器
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    optimizer_top = torch.optim.Adam(model.dn1.parameters(), lr=args.lr_top)
    optimizer_main = torch.optim.Adam(model.dn2.parameters(), lr=args.lr_main)
    optimizer_other = torch.optim.Adam(other_params, lr=args.lr_other)
    scheduler_top = MultiStepLR(optimizer_top, args.milestones_top, args.gamma)
    scheduler_main = MultiStepLR(optimizer_main, args.milestones_main, args.gamma)
    scheduler_other = MultiStepLR(optimizer_other, args.milestones_other, args.gamma)
    # scheduler_top = ReduceLROnPlateau(optimizer_top, mode='min', factor=args.gamma,
    #                                   patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                   min_lr=0, eps=1e-08)
    # scheduler_main = ReduceLROnPlateau(optimizer_main, mode='min', factor=args.gamma,
    #                                    patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                    min_lr=0, eps=1e-08)
    # scheduler_other = ReduceLROnPlateau(optimizer_other, mode='min', factor=args.gamma,
    #                                     patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                     min_lr=0, eps=1e-08)

    # 保存每个epoch后的Accuracy Loss Val_Accuracy
    Accuracy_top, Loss_top, Val_Loss_top, Val_Accuracy_top, Test_Loss_top, Test_Accuracy_top,  = [], [], [], [], [], []
    BEST_VAL_ACC_top, BEST_epoch_top = 0., -1

    Accuracy_main, Loss_main, Val_Loss_main, Val_Accuracy_main, Test_Loss_main, Test_Accuracy_main = [], [], [], [], [], []
    BEST_VAL_ACC_main, BEST_epoch_main = 0., -1

    Accuracy_fusion, Loss_fusion, Val_Loss_fusion, Val_Accuracy_fusion, Test_Loss_fusion, Test_Accuracy_fusion = [], [], [], [], [], []
    BEST_VAL_ACC_fusion, BEST_epoch_fusion = 0., -1

    # 训练
    for epoch in range(epochs):
        train_loss_top, train_corrects_top, run_accuracy_top, run_loss_top, total_top = 0., 0., 0., 0., 0.
        train_loss_main, train_corrects_main, run_accuracy_main, run_loss_main, total_main = 0., 0., 0., 0., 0.
        train_loss_fusion, train_corrects_fusion, run_accuracy_fusion, run_loss_fusion, total_fusion = 0., 0., 0., 0., 0.
        model.train()
        consistency_weight = get_current_consistency_weight(epoch)
        data_main = iter(data_loaders[1][0])
        for i, data in enumerate(data_loaders[0][0], 1):
            state_top, state_fusion = 0, 0
            images_top, labels_top = data
            images_top = images_top.to(device)
            labels_top = labels_top.to(device)

            (images_main, labels_main) = next(data_main)
            images_main = images_main.to(device)
            labels_main = labels_main.to(device)

            problems_label = np.array([], dtype=int)
            for idx in range(labels_top.shape[0]):
                problems_label = np.append(problems_label, get_problem(labels_top[idx], labels_main[idx]))
            problems_label_torch = torch.from_numpy(problems_label).to(device)

            # 经典四步
            optimizer_top.zero_grad()
            optimizer_main.zero_grad()
            optimizer_other.zero_grad()

            outs_fusion, outs_top, outs_main = model(images_top, images_main)

            target_onehot = torch.autograd.Variable(
                (torch.zeros(images_top.size()[0], 4).cuda()).scatter_(1, problems_label_torch.view(
                    problems_label_torch.size()[0], 1).cuda(), 1))

            main_label = dist_s_label(target_onehot, outs_main.detach())
            top_label = dist_s_label(target_onehot, outs_top.detach())
            fusion_label = dist_s_label(target_onehot, outs_fusion.detach())

            # norm
            pmain_ptop = dist_s_t(outs_top.detach(), outs_main.detach(), 1)
            pmain_pfusion = dist_s_t(outs_fusion.detach(), outs_main.detach(), 1)
            ptop_pfusion = dist_s_t(outs_fusion.detach(), outs_top.detach(), 1)

            epsilon = torch.exp(-1 * top_label / (main_label + top_label))
            delta = main_label - epsilon * top_label
            epsilon1 = torch.exp(-1 * fusion_label / (main_label + fusion_label))
            delta1 = main_label - epsilon1 * fusion_label
            epsilon2 = torch.exp(-1 * fusion_label / (top_label + fusion_label))
            delta2 = top_label - epsilon2 * fusion_label

            if (pmain_ptop > delta and top_label < main_label) and (pmain_pfusion > delta1 and fusion_label < main_label) and (ptop_pfusion > delta2 and fusion_label < top_label):
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main
                loss.backward()
                optimizer_main.step()
            elif (pmain_ptop > delta and top_label < main_label) and (pmain_pfusion > delta1 and fusion_label < main_label):
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main
                loss.backward()
                optimizer_main.step()
            elif (pmain_ptop > delta and top_label < main_label) and (ptop_pfusion > delta2 and fusion_label < top_label):
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main
                loss.backward()
                optimizer_main.step()
            elif (pmain_pfusion > delta1 and fusion_label < main_label) and (ptop_pfusion > delta2 and fusion_label < top_label):
                state_top = 1
                loss_top = criterion(outs_top, problems_label_torch) + \
                           (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main + loss_top
                loss.backward()
                optimizer_main.step()
                optimizer_top.step()
            elif (pmain_ptop > delta and top_label < main_label):
                state_fusion = 1
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss_fusion = criterion(outs_fusion, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_fusion) * 1 + kl_div(outs_main.detach(), outs_fusion)) * 1
                loss = loss_main + loss_fusion
                loss.backward()
                optimizer_main.step()
                optimizer_other.step()
            elif (pmain_pfusion > delta1 and fusion_label < main_label):
                state_top = 1
                loss_top = criterion(outs_top, problems_label_torch) + \
                           (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main + loss_top
                loss.backward()
                optimizer_main.step()
                optimizer_top.step()
            elif (ptop_pfusion > delta2 and fusion_label < top_label):
                state_top = 1
                loss_top = criterion(outs_top, problems_label_torch) + \
                           (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss = loss_main + loss_top
                loss.backward()
                optimizer_main.step()
                optimizer_top.step()
            else:
                state_top = 1
                state_fusion = 1
                loss_top = criterion(outs_top, problems_label_torch) + \
                           (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
                loss_main = criterion(outs_main, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
                loss_fusion = criterion(outs_fusion, problems_label_torch) + \
                            (kl_div(outs_top.detach(), outs_fusion) * 1 + kl_div(outs_main.detach(), outs_fusion)) * 1
                loss = loss_top + loss_main + loss_fusion
                loss.backward()
                optimizer_top.step()
                optimizer_main.step()
                optimizer_other.step()
            # if (pmain_ptop > delta and top_label < main_label) and (pmain_pfusion > delta1 and fusion_label < main_label):
            #     loss_main = criterion(outs_main, problems_label_torch) + \
            #                 (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
            #     loss = loss_main
            #     loss.backward()
            #     optimizer_main.step()
            # elif pmain_ptop > delta and top_label < main_label:
            #     state_fusion = 1
            #     loss_main = criterion(outs_main, problems_label_torch) + \
            #                 (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
            #     loss_fusion = criterion(outs_fusion, problems_label_torch)
            #     loss = loss_main + loss_fusion
            #     loss.backward()
            #     optimizer_main.step()
            #     optimizer_other.step()
            # elif pmain_pfusion > delta1 and fusion_label < main_label:
            #     state_top = 1
            #     loss_top = criterion(outs_top, problems_label_torch) + \
            #                (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
            #     loss_main = criterion(outs_main, problems_label_torch) + \
            #                 (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
            #     loss = loss_main + loss_top
            #     loss.backward()
            #     optimizer_main.step()
            #     optimizer_top.step()
            # else:
            #     state_top = 1
            #     state_fusion = 1
            #     loss_top = criterion(outs_top, problems_label_torch) + \
            #                (kl_div(outs_main.detach(), outs_top) * 1 + kl_div(outs_fusion.detach(), outs_top)) * 1
            #     loss_main = criterion(outs_main, problems_label_torch) + \
            #                 (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
            #     loss_fusion = criterion(outs_fusion, problems_label_torch)
            #     loss = loss_top + loss_main + loss_fusion
            #     loss.backward()
            #     optimizer_top.step()
            #     optimizer_main.step()
            #     optimizer_other.step()

            total_top += labels_top.size(0)
            run_loss_top += loss_top.item()
            train_loss_top += loss_top.item()
            prediction_top = torch.max(outs_top, dim=1)[1]
            run_accuracy_top += (prediction_top == problems_label_torch.to(device)).sum().item()
            train_corrects_top += (prediction_top == problems_label_torch.to(device)).sum().item()
            rate_top = i / len(data_loaders[0][0])
            a_top = "*" * int(rate_top * 50)
            b_top = "." * int((1 - rate_top) * 50)

            print(
                "\rTop train loss: {:^3.0f}%[{}->{}]{:.4f}||{:.4f}||{}\n".format(int(rate_top * 100), a_top,
                                                                                 b_top,
                                                                                 run_loss_top / i,
                                                                                 run_accuracy_top / total_top, state_top),
                end="")

            # 输出状态
            total_main += labels_main.size(0)
            run_loss_main += loss_main.item()
            train_loss_main += loss_main.item()
            prediction_main = torch.max(outs_main, dim=1)[1]
            run_accuracy_main += (prediction_main == problems_label_torch.to(device)).sum().item()
            train_corrects_main += (prediction_main == problems_label_torch.to(device)).sum().item()
            rate_main = i / len(data_loaders[0][0])
            a_main = "*" * int(rate_main * 50)
            b_main = "." * int((1 - rate_main) * 50)

            print("\rMain train loss: {:^3.0f}%[{}->{}]{:.4f}||{:.4f}\n".format(int(rate_main * 100),
                                                                                a_main, b_main,
                                                                                run_loss_main / i,
                                                                                run_accuracy_main / total_main), end="")

            # 输出状态
            total_fusion += labels_top.size(0)
            run_loss_fusion += loss_fusion.item()
            train_loss_fusion += loss_fusion.item()
            prediction_fusion = torch.max(outs_fusion, dim=1)[1]
            run_accuracy_fusion += (prediction_fusion == problems_label_torch.to(device)).sum().item()
            train_corrects_fusion += (prediction_fusion == problems_label_torch.to(device)).sum().item()
            rate_fusion = i / len(data_loaders[0][0])
            a_fusion = "*" * int(rate_fusion * 50)
            b_fusion = "." * int((1 - rate_fusion) * 50)

            print("\rFusion train loss: {:^3.0f}%[{}->{}]{:.4f}||{:.4f}||{}\n".format(int(rate_fusion * 100),
                                                                                  a_fusion, b_fusion,
                                                                                  run_loss_fusion / i,
                                                                                  run_accuracy_fusion / total_fusion, state_fusion),
                  end="")

        epoch_acc_top = 100 * train_corrects_top / total_top
        epoch_loss_top = train_loss_top / total_top * batchsize
        Loss_top.append(epoch_loss_top)
        Accuracy_top.append(epoch_acc_top)

        epoch_acc_main = 100 * train_corrects_main / total_main
        epoch_loss_main = train_loss_main / total_main * batchsize
        Loss_main.append(epoch_loss_main)
        Accuracy_main.append(epoch_acc_main)

        epoch_acc_fusion = 100 * train_corrects_fusion / total_fusion
        epoch_loss_fusion = train_loss_fusion / total_fusion * batchsize
        Loss_fusion.append(epoch_loss_fusion)
        Accuracy_fusion.append(epoch_acc_fusion)

        # 验证
        model.eval()
        with torch.no_grad():
            accuracy_top, total_val_top, val_loss_top = 0., 0, 0.
            accuracy_main, total_val_main, val_loss_main = 0., 0, 0.
            accuracy_fusion, total_val_fusion, val_loss_fusion = 0., 0, 0.
            data_main_val = iter(data_loaders[1][1])
            for data in data_loaders[0][1]:
                images_top, labels_top = data
                images_top = images_top.to(device)
                labels_top = labels_top.to(device)

                (images_main, labels_main) = next(data_main_val)
                images_main = images_main.to(device)
                labels_main = labels_main.to(device)

                problems_label = np.array([], dtype=int)
                for idx in range(labels_top.shape[0]):
                    problems_label = np.append(problems_label, get_problem(labels_top[idx], labels_main[idx]))
                problems_label_torch = torch.from_numpy(problems_label).to(device)

                out_fusion, out_top, out_main = model(images_top, images_main)

                _, prediction_top = torch.max(out_top, 1)
                total_val_top += problems_label_torch.size(0)
                loss_top = criterion(out_top, problems_label_torch) + \
                           (kl_div(out_main.detach(), out_top) * 0.9 + kl_div(out_fusion.detach(), out_top)) * 0.1
                val_loss_top += loss_top.item()
                accuracy_top += (prediction_top == problems_label_torch.to(device)).sum().item()

                _, prediction_main = torch.max(out_main, 1)
                total_val_main += problems_label_torch.size(0)
                loss_main = criterion(out_main, problems_label_torch) + \
                            (kl_div(out_top.detach(), out_main) * 0.9 + kl_div(out_fusion.detach(), out_main)) * 0.1
                val_loss_main += loss_main.item()
                accuracy_main += (prediction_main == problems_label_torch.to(device)).sum().item()

                _, prediction_fusion = torch.max(out_fusion, 1)
                total_val_fusion += problems_label_torch.size(0)
                loss_fusion = criterion(out_fusion, problems_label_torch)
                val_loss_fusion += loss_fusion.item()
                accuracy_fusion += (prediction_fusion == problems_label_torch.to(device)).sum().item()

            val_acc_top = 100. * accuracy_top / total_val_top
            val_acc_main = 100. * accuracy_main / total_val_main
            val_acc_fusion = 100. * accuracy_fusion / total_val_fusion

            print(
                'Top epoch：{}/{} || lr: {:.6f} || Train_Loss: {:.4f} || Epoch_ACC: {:.4f}% || Val_Loss: {:.4f} '
                'Val_ACC: {:.8f}% || \n'.format(epoch + 1, epochs,
                                                optimizer_top.param_groups[0][
                                                    'lr'],
                                                epoch_loss_top,
                                                epoch_acc_top,
                                                val_loss_top / total_val_top * batchsize,
                                                val_acc_top))
            print(
                'Main epoch：{}/{} || lr: {:.6f} || Train_Loss: {:.4f} || Epoch_ACC: {:.4f}% || Val_Loss: {:.4f} '
                'Val_ACC: {:.8f}% || \n'.format(epoch + 1, epochs,
                                                optimizer_main.param_groups[
                                                    0]['lr'],
                                                epoch_loss_main,
                                                epoch_acc_main,
                                                val_loss_main / total_val_main * batchsize,
                                                val_acc_main))
            print(
                'Fusion epoch：{}/{} || lr: {:.6f} || Train_Loss: {:.4f} || Epoch_ACC: {:.4f}% || Val_Loss: {:.4f} '
                'Val_ACC: {:.8f}% || \n'.format(epoch + 1, epochs,
                                                optimizer_other.param_groups[
                                                    0]['lr'],
                                                epoch_loss_fusion,
                                                epoch_acc_fusion,
                                                val_loss_fusion / total_val_fusion * batchsize,
                                                val_acc_fusion))

            Val_Loss_top.append(val_loss_top / total_val_top * batchsize)
            Val_Accuracy_top.append(val_acc_top)
            # if val_acc_top > BEST_VAL_ACC_top:
            #     BEST_VAL_ACC_top = val_acc_top
            #     if not os.path.isdir('checkpoint'):
            #         os.mkdir('checkpoint')
            #     torch.save(model_T.state_dict(),
            #                './checkpoint/' + exp_name + '_' + exp_time + type + '_top.pth')
            #     BEST_epoch_top = epoch + 1
            # # scheduler_top.step()
            #
            Val_Loss_main.append(val_loss_main / total_val_main * batchsize)
            Val_Accuracy_main.append(val_acc_main)
            # if val_acc_main > BEST_VAL_ACC_main:
            #     BEST_VAL_ACC_main = val_acc_main
            #     if not os.path.isdir('checkpoint'):
            #         os.mkdir('checkpoint')
            #     torch.save(model_S.state_dict(),
            #                './checkpoint/' + exp_name + '_' + exp_time + type + '_main.pth')
            #     BEST_epoch_main = epoch + 1
            # # scheduler_main.step()

            Val_Loss_fusion.append(val_loss_fusion / total_val_fusion * batchsize)
            Val_Accuracy_fusion.append(val_acc_fusion)
            if val_acc_fusion > BEST_VAL_ACC_fusion:
                BEST_VAL_ACC_fusion = val_acc_fusion
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(model.state_dict(),
                           './checkpoint/' + exp_name + '_' + exp_time + type + '_fusion.pth')
                BEST_epoch_fusion = epoch + 1
            # scheduler_top.step(val_loss_top / total_val_top * batchsize)
            # scheduler_main.step(val_loss_main / total_val_main * batchsize)
            # scheduler_other.step(val_loss_fusion / total_val_fusion * batchsize)
            scheduler_top.step()
            scheduler_main.step()
            scheduler_other.step()

        # print('Now the highest top model is epoch {}, val acc is {}'.format(BEST_epoch_top, BEST_VAL_ACC_top))
        #
        # print('Now the highest main model is epoch {}, val acc is {}'.format(BEST_epoch_main, BEST_VAL_ACC_main))
        print('Now the highest model is epoch {}, val acc is {}'.format(BEST_epoch_fusion, BEST_VAL_ACC_fusion))

    draw_acc_loss(Loss_top, Val_Loss_top, Accuracy_top, Val_Accuracy_top, args.pic_name1)
    draw_acc_loss(Loss_main, Val_Loss_main, Accuracy_main, Val_Accuracy_main, args.pic_name2)
    draw_acc_loss(Loss_fusion, Val_Loss_fusion, Accuracy_fusion, Val_Accuracy_fusion, args.pic_name3)

    model.load_state_dict(torch.load('./checkpoint/' + exp_name + '_' + exp_time + type + '_fusion.pth'))
    model.eval()

    loss_total_top = 0
    predict_all_top = np.array([], dtype=int)
    labels_all_top = np.array([], dtype=int)

    loss_total_main = 0
    predict_all_main = np.array([], dtype=int)
    labels_all_main = np.array([], dtype=int)

    loss_total_fusion = 0
    predict_all_fusion = np.array([], dtype=int)
    labels_all_fusion = np.array([], dtype=int)
    with torch.no_grad():
        data_main_test = iter(data_loaders[1][2])
        for step, (x2, y2) in enumerate(data_loaders[0][2], 1):
            x2 = x2.to(device)
            y2 = y2.to(device)

            (images_main, labels_main) = next(data_main_test)
            images_main = images_main.to(device)
            labels_main = labels_main.to(device)

            problems_label = np.array([], dtype=int)
            for idx in range(y2.shape[0]):
                problems_label = np.append(problems_label, get_problem(y2[idx], labels_main[idx]))
            problems_label_torch2 = torch.from_numpy(problems_label).to(device)

            outputs_fusion, outputs_top, outputs_main = model(x2, images_main)

            loss_top = F.cross_entropy(outputs_top, problems_label_torch2)
            loss_total_top += loss_top
            problems_label_torch2 = problems_label_torch2.data.cpu().numpy()
            predic_top = torch.max(outputs_top.data, 1)[1].cpu().numpy()
            labels_all_top = np.append(labels_all_top, problems_label_torch2)
            predict_all_top = np.append(predict_all_top, predic_top)

            problems_label_torch2 = torch.from_numpy(problems_label).to(device)

            loss_main = F.cross_entropy(outputs_main, problems_label_torch2)
            loss_total_main += loss_main
            problems_label_torch2 = problems_label_torch2.data.cpu().numpy()
            predic_main = torch.max(outputs_main.data, 1)[1].cpu().numpy()
            labels_all_main = np.append(labels_all_main, problems_label_torch2)
            predict_all_main = np.append(predict_all_main, predic_main)

            problems_label_torch2 = torch.from_numpy(problems_label).to(device)

            loss_fusion = F.cross_entropy(outputs_fusion, problems_label_torch2)
            loss_total_fusion += loss_fusion
            problems_label_torch2 = problems_label_torch2.data.cpu().numpy()
            predic_fusion = torch.max(outputs_fusion.data, 1)[1].cpu().numpy()
            labels_all_fusion = np.append(labels_all_fusion, problems_label_torch2)
            predict_all_fusion = np.append(predict_all_fusion, predic_fusion)

    return BEST_VAL_ACC_top, BEST_VAL_ACC_main, loss_total_top / len(data_loaders[0][2]), loss_total_main / len(
        data_loaders[0][
            2]), labels_all_top, labels_all_main, labels_all_fusion, predict_all_top, predict_all_main, predict_all_fusion


for times in range(5):
    data_dir = args.data_dir
    batchsize = args.batch_size

    top_list, main_list = subset_random_sampler(data_dir, batchsize=batchsize, ratio=args.ratio, seed=args.seed,
                                                is_upsample=args.DA,
                                                aug_times=args.aug_times)  # is_upsample=True, aug_times=3
    # top_train_dataloader, top_val_dataloader, top_test_dataloader = top_list[0], top_list[1], top_list[2]
    print("Top view data: ", top_list[3], top_list[4], top_list[5])
    # main_train_dataloader, main_val_dataloader, main_test_dataloader = main_list[0], main_list[1], main_list[2]
    print("Main view data: ", main_list[3], main_list[4], main_list[5])

    # 加载预训练模型
    model = densenet121(num_class=args.num_classes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BEST_VAL_ACC_TOP, BEST_VAL_ACC_Main, loss_top, loss_main, final_label_top, final_label_main, final_label_fusion, final_pre_top, final_pre_main, final_pre_fusion = train(
        model, device, args.epoch, [top_list, main_list], args.type)

    acc_top = metrics.accuracy_score(final_label_top, final_pre_top)
    report_top = metrics.classification_report(final_label_top, final_pre_top,
                                               target_names=['0_空洞', '1_层间脱空', '2_裂缝', '3_正常'],
                                               digits=4)
    print('Top Test Acc: {}'.format(acc_top))
    print("Precision, Recall and F1-Score...")
    print(report_top)

    acc_main = metrics.accuracy_score(final_label_main, final_pre_main)
    report_main = metrics.classification_report(final_label_main, final_pre_main,
                                                target_names=['0_空洞', '1_层间脱空', '2_裂缝', '3_正常'],
                                                digits=4)
    print('Main Test Acc: {}'.format(acc_main))
    print("Precision, Recall and F1-Score...")
    print(report_main)

    acc_voting = metrics.accuracy_score(final_label_fusion, final_pre_fusion)
    report_voting = metrics.classification_report(final_label_fusion, final_pre_fusion,
                                                  target_names=['0_空洞', '1_层间脱空', '2_裂缝', '3_正常'], digits=4)
    print('Voting Test Acc: {}'.format(acc_voting))
    print("Precision, Recall and F1-Score...")
    print(report_voting)

    print(final_label_fusion)
    print(final_pre_fusion)


    accuracy = metrics.accuracy_score(final_label_fusion, final_pre_fusion)
    precision = metrics.precision_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro')
    recall = metrics.recall_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro')
    f1_score = metrics.f1_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro')
    print("accuracy：%.6f" % metrics.accuracy_score(final_label_fusion, final_pre_fusion))
    print("precision：%.6f" % metrics.precision_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro'))
    print("recall：%.6f" % metrics.recall_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro'))
    print("f1-score：%.6f" % metrics.f1_score(final_label_fusion, final_pre_fusion, labels=[0, 1, 2], average='macro'))

    title = ['实验', '优化器', 'lr', 'top_test_report', 'main_test_report', 'test_report', 'TOP_TEST_ACC', 'MAIN_TEST_ACC',
             'TEST_ACC', 'accuracy', 'precision', 'recall', 'f1-score', 'channel',
             'epoch', 'batch_size',
             'image_size', 'data_size', 'early_stop', 'up_sampling', 'transforms', 'T', 'w_top', 'w_main', 'momentum',
             'weight-decay', 'gamma', 'milestones_top', 'milestones_main']
    rows = [
        [exp_name, 'Adam', args.lr, report_top, report_main, report_voting, acc_top, acc_main, acc_voting, accuracy, precision, recall, f1_score, 'RGB',
         args.epoch,
         args.batch_size,
         '224:224',
         str(top_list[3]) + ':' + str(top_list[4]) + ":" + str(top_list[5]), False, args.DA, 'color,ver', args.T,
         args.w_top, args.w_main, args.momentum, args.weight_decay, args.gamma, args.milestones_top,
         args.milestones_main]]

    with open(exp_path + '/' + exp_name + '.csv', 'a', newline='') as csv_file:
        # 获取一个csv对象进行内容写入
        writer = csv.writer(csv_file)
        # writer.writerow(title)
        for row in rows:
            # writerow 写入一行数据
            writer.writerow(row)
        # 写入多行
        # writer.writerows(rows)
        writer1 = csv.writer(csv_file)
