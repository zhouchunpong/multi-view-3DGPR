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
from torch.optim.lr_scheduler import MultiStepLR
from data import get_problem, subset_random_sampler, subset_random_sampler_end_to_end
from models.AlexNet import MyAlexNet
import csv
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', type=str, nargs='+', default=['AlexNet'])
parser.add_argument('--type', type=str, default='SingleTopView')
parser.add_argument('--version', type=str, default='base')

parser.add_argument('--data_dir', type=str, default='./radar_data')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--epoch', type=int, default=160)
parser.add_argument('--ratio', type=int, default=[0.5, 0.2, 0.3])

parser.add_argument('--DA', type=bool, default=True)
parser.add_argument('--aug_times', type=int, default=0)

parser.add_argument('--T', type=float, default=4.0)  # temperature
parser.add_argument('--w_top', type=float, default=0.3)  # weight for ce and kl
parser.add_argument('--w_main', type=float, default=0.3)  # weight for ce and kl

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--milestones', type=int, nargs='+', default=[60, 80])

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu-id', type=int, default=4)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--pic_name1', type=str, default='top')
parser.add_argument('--pic_name2', type=str, default='main')

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


def train(model, device, epochs, data_loaders, type):
    model.to(device)
    # 定义loss function和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5
    # scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)

    # 保存每个epoch后的Accuracy Loss Val_Accuracy
    Accuracy, Loss, Val_Loss, Val_Accuracy, Test_Loss, Test_Accuracy, = [], [], [], [], [], []
    BEST_VAL_ACC, BEST_epoch = 0., -1

    # 训练
    for epoch in range(epochs):
        train_loss, train_corrects, run_accuracy, run_loss, total = 0., 0., 0., 0., 0.
        model.train()
        data_main = iter(data_loaders[1][0])
        for i, data in enumerate(data_loaders[0][0], 1):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            (images_main, labels_main) = next(data_main)
            images_main = images_main.to(device)
            labels_main = labels_main.to(device)

            problems_label = np.array([], dtype=int)
            for idx in range(labels.shape[0]):
                problems_label = np.append(problems_label, get_problem(labels[idx], labels_main[idx]))
            problems_label_torch = torch.from_numpy(problems_label).to(device)

            # 经典四步
            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, problems_label_torch)
            loss.backward()
            optimizer.step()
            # 输出状态
            total += labels.size(0)
            run_loss += loss.item()
            train_loss += loss.item()
            prediction = torch.max(outs, dim=1)[1]
            run_accuracy += (prediction == problems_label_torch.to(device)).sum().item()
            train_corrects += (prediction == problems_label_torch.to(device)).sum().item()
            rate = i / len(data_loaders[0][0])
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)

            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}||{:.4f}".format(int(rate * 100), a, b, run_loss / i,
                                                                         run_accuracy / total), end="")

        epoch_acc = 100 * train_corrects / total
        epoch_loss = train_loss / total * batchsize
        Loss.append(epoch_loss)
        Accuracy.append(epoch_acc)

        # scheduler.step(epoch_loss)

        # 验证
        val_acc = 0.

        model.eval()
        print()
        with torch.no_grad():
            accuracy = 0.
            total_val = 0
            val_loss = 0.
            data_main_val = iter(data_loaders[1][1])
            for data in data_loaders[0][1]:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                (images_main, labels_main) = next(data_main_val)
                images_main = images_main.to(device)
                labels_main = labels_main.to(device)

                problems_label = np.array([], dtype=int)
                for idx in range(labels.shape[0]):
                    problems_label = np.append(problems_label, get_problem(labels[idx], labels_main[idx]))
                problems_label_torch = torch.from_numpy(problems_label).to(device)

                out = model(images)
                _, prediction = torch.max(out, 1)
                total_val += problems_label_torch.size(0)
                loss = criterion(out, problems_label_torch)
                val_loss += loss.item()
                accuracy += (prediction == problems_label_torch.to(device)).sum().item()

            val_acc = 100. * accuracy / total_val
            # scheduler.step(val_acc)

            print(
                'epoch：{}/{} || lr: {:.4f} || Train_Loss: {:.4f} || Epoch_ACC: {:.4f}% || Val_Loss: {:.4f} '
                'Val_ACC: {:.8f}% \n'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], epoch_loss,
                                             epoch_acc,
                                             val_loss / total_val * batchsize, val_acc))
            Val_Loss.append(val_loss / total_val * batchsize)
            Val_Accuracy.append(val_acc)
            if val_acc > BEST_VAL_ACC:
                BEST_VAL_ACC = val_acc
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(model.state_dict(), './checkpoint/densenet121_top_sep_radar_' + type + '.pth')
                BEST_epoch = epoch + 1
            # scheduler.step()
        print('Now the highest model is epoch {}, val acc is {}'.format(BEST_epoch, BEST_VAL_ACC))

    draw_acc_loss(Loss, Val_Loss, Accuracy, Val_Accuracy, args.pic_name1)
    model.load_state_dict(torch.load('./checkpoint/densenet121_top_sep_radar_' + type + '.pth'))
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
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

            outputs = model(x2)
            loss = F.cross_entropy(outputs, problems_label_torch2)
            loss_total += loss
            problems_label_torch2 = problems_label_torch2.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, problems_label_torch2)
            predict_all = np.append(predict_all, predic)

    return BEST_VAL_ACC, loss_total / len(data_loaders[0][2]), labels_all, predict_all


for times in range(10):
    data_dir = args.data_dir
    batchsize = args.batch_size

    top_list, main_list = subset_random_sampler(data_dir, batchsize=batchsize, ratio=args.ratio, seed=args.seed,
                                                is_upsample=args.DA, aug_times=args.aug_times)  # is_upsample=True, aug_times=3
    # top_train_dataloader, top_val_dataloader, top_test_dataloader = top_list[0], top_list[1], top_list[2]
    print("Top view data: ", top_list[3], top_list[4], top_list[5])
    # main_train_dataloader, main_val_dataloader, main_test_dataloader = main_list[0], main_list[1], main_list[2]
    print("Main view data: ", main_list[3], main_list[4], main_list[5])

    # 加载预训练模型
    model_top = MyAlexNet(num_classes=args.num_classes)
    # model_main = densenet121(num_class=3)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Top_BEST_VAL_ACC, loss_top, labels_all_top, predict_all_top = train(model_top, device, args.epoch,
                                                                        [top_list, main_list],
                                                                        args.type)

    acc_top = metrics.accuracy_score(labels_all_top, predict_all_top)
    report_top = metrics.classification_report(labels_all_top, predict_all_top,
                                               target_names=['0_空洞', '1_层间脱空', '2_裂缝', '3_正常'], digits=4)

    print('Top Test Loss: {0:>5.2}, Top Test Acc: {1:>6.2%}'.format(loss_top, acc_top))
    print("Precision, Recall and F1-Score...")
    print(report_top)

    accuracy = metrics.accuracy_score(labels_all_top, predict_all_top)
    precision = metrics.precision_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro')
    recall = metrics.recall_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro')
    f1_score = metrics.f1_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro')
    print("accuracy：%.6f" % metrics.accuracy_score(labels_all_top, predict_all_top))
    print("precision：%.6f" % metrics.precision_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro'))
    print("recall：%.6f" % metrics.recall_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro'))
    print("f1-score：%.6f" % metrics.f1_score(labels_all_top, predict_all_top, labels=[0, 1, 2], average='macro'))


    title = ['实验', '优化器', 'lr', 'top_test_report', 'TOP_BEST_VAL_ACC',
             'TOP_TEST_ACC', 'accuracy', 'precision', 'recall', 'f1-score', 'channel', 'epoch', 'batch_size',
             'image_size', 'data_size', 'early_stop', 'up_sampling', 'transforms', 'T', 'w_top', 'w_main', 'momentum', 'weight-decay', 'gamma', 'milestones']
    rows = [
        [exp_name, 'Adam', args.lr, report_top, Top_BEST_VAL_ACC, acc_top, accuracy, precision, recall, f1_score, 'RGB', args.epoch, args.batch_size,
         '224:224',
         str(top_list[3]) + ':' + str(top_list[4]) + ":" + str(top_list[5]), False, True, 'color,ver', args.T, args.w_top, args.w_main, args.momentum, args.weight_decay, args.gamma, args.milestones]]

    with open(exp_path + '/' + exp_name + '.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)
        writer1 = csv.writer(csv_file)
