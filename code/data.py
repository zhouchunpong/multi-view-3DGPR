import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, DataLoader, BatchSampler, SequentialSampler
from sampler import SubsetRandomSampler
import numpy as np
from torchvision.datasets import ImageFolder
from data_augmention_loader_ori import augmention_dataset
import os
import PIL.Image as Image
import torch.nn.functional as F
from torchvision.transforms import functional as F
from collections import Counter
import torchvision


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """

    def __init__(self, mean, sigma):

        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 将图片灰度标准化
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        # 产生高斯 noise
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out*255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('RGB')


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        #mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[:, x1:x2, y1:y2] = 0.
            #mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        # img = img * mask

        return img


class HidePatch(object):
    def __init__(self, grid_sizes, hide_prob=0.5):
        self.hide_prob = hide_prob
        self.grid_sizes = grid_sizes

    def __call__(self, img):
        # get width and height of the image
        s = F.get_image_size(img)
        wd = s[0]
        ht = s[1]

        grid_sizes = self.grid_sizes
        grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]

        # hide the patches
        if grid_size > 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= self.hide_prob:
                        img[:, x:x_end, y:y_end] = 0

        return img

def up_sample(indices, classes):
    # [103, 13, 171, 120]
    max_num = max(classes)
    up_indices = indices
    up_classes = []
    for i in range(len(classes)):
        up_classes.append(classes[i] + up_classes[i - 1]) if up_classes else up_classes.append(classes[i])
        if classes[i] < max_num:
            index = np.random.randint(classes[i], size=(max_num - classes[i]))  # 随机给定上采样取出样本的序号
            for item in index:
                up_indices.append(indices[item + up_classes[i - 1]]) if i > 0 else up_indices.append(indices[item])
    return up_indices


def random_split(data_dir, batchsize, ratio=None, aug_times=0):
    if ratio is None:
        ratio = [0.6, 0.2, 0.2]
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 进行随机水平翻转
        transforms.RandomVerticalFlip(),  # 进行随机竖直翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_dir, transform=transformer)

    # 随机数据划分
    num_sample_train = int(len(dataset) * ratio[0])
    num_sample_val = int(len(dataset) * ratio[1])
    num_sample_test = len(dataset) - num_sample_train - num_sample_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_sample_train, num_sample_val,
                                                                                       num_sample_test])

    if aug_times > 0:
        train_img_list = [train_dataset.dataset.imgs[train_dataset.indices[i]][0] for i in
                          range(len(train_dataset.indices))]
        train_dset = augmention_dataset(sub_dir=data_dir, class_to_idx=None, image_list=train_img_list,
                                        transform=train_trans)
        train_dset.shuffle_data(True)
        train_dset.setmode(2)
        train_dset.maketraindata(aug_times)
        train_dset.shuffle_data(True)

        train_dataloader = DataLoader(train_dset, batch_size=batchsize, shuffle=False, num_workers=0)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    train_num, val_num, test_num = len(train_dataloader.dataset), len(val_dataloader.dataset), len(
        test_dataloader.dataset)

    return train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num


# train_indices, val_indices, test_indices = get_random_indices(dataset_topview.targets, ratio=ratio,
#                                                                       is_upsample=is_upsample)
def get_random_indices(dataset_labels, seed, ratio, is_upsample=False):
    # create indices depended on labels
    if ratio is None:
        ratio = [0.6, 0.2, 0.2]

    shuffle_dataset = True
    random_seed = seed

    # Creating data indices for training and validation splits:
    labels = dataset_labels
    label_set = set(labels)
    train_indices, val_indices, test_indices = [], [], []
    class_num, train_classes = [], []
    for item in label_set:
        temp_num = labels.count(item)
        if item > 0:
            class_num.append(temp_num + class_num[item - 1])
            indices = list(range(class_num[item - 1], class_num[item]))
        else:
            class_num.append(temp_num)
            indices = list(range(class_num[item]))
        num_sample_train = int(np.floor(ratio[0] * temp_num))
        num_sample_val = int(np.floor(ratio[1] * temp_num))
        train_classes.append(num_sample_train)
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices.extend(indices[:num_sample_train])
        val_indices.extend(indices[num_sample_train:(num_sample_train + num_sample_val)])
        test_indices.extend(indices[(num_sample_train + num_sample_val):])
    # up sampling
    if is_upsample:
        train_indices = up_sample(train_indices, train_classes)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def create_dataloader_combine(dataset, batchsize, train_indices, val_indices, test_indices, aug_times=0):
    # Creating PT data samplers and loaders:
    train_trans = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 进行随机水平翻转
        # transforms.RandomVerticalFlip(),  # 进行随机竖直翻转
        # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # valid_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size=batchsize, drop_last=False)
    # test_sampler = BatchSampler(SubsetRandomSampler(test_indices), batch_size=batchsize, drop_last=False)
    valid_sampler = SequentialSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    print("train indices: ", train_indices)
    print("val indices: ", val_indices)
    print("test indices: ", test_indices)

    if aug_times > 0:
        train_img_list = [dataset.imgs[train_indices[i]][0] for i in range(len(train_indices))]
        train_dset = augmention_dataset(sub_dir=data_dir, class_to_idx=None, image_list=train_img_list,
                                        transform=train_trans)
        # train_dset.shuffle_data(True)
        train_dset.setmode(2)
        train_dset.maketraindata(aug_times)
        # train_dset.shuffle_data(True)
        train_dataloader = DataLoader(train_dset, batch_size=batchsize, shuffle=False, num_workers=0)
        train_num = len(train_dataloader.dataset)
    else:
        # train_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size=batchsize, drop_last=False)
        # train_dataloader = DataLoader(dataset, batch_sampler=train_sampler)
        train_sampler = SubsetRandomSampler(train_indices)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batchsize, drop_last=False,
                                      num_workers=0)
        train_num = len(train_dataloader.sampler)

    # val_dataloader = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=0)
    # test_dataloader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=0)

    val_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=batchsize, num_workers=0)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=batchsize, num_workers=0)

    val_num, test_num = len(val_dataloader.sampler), len(test_dataloader.sampler)

    return [train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num]


# topview = create_dataloader(dataset_topview, os.path.join(data_dir, "0_topview"), batchsize, train_indices,
#                                     val_indices, test_indices, aug_times=aug_times)
def create_dataloader(dataset, data_dir, batchsize, train_indices, val_indices, test_indices, aug_times=0):
    # Creating PT data samplers and loaders:
    noisef = 1-0.2
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 进行随机水平翻转
        transforms.RandomVerticalFlip(),  # 进行随机竖直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transforms.CenterCrop((280, 220)),
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        # AddPepperNoise(noisef, p=0.9),  # 加椒盐噪声
        # Gaussian_noise(0, noisef),  # 加高斯噪声
        # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.25, scale=(0.02, 0.4), value='random'),
        # transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1) can not use
    ])
    # train_trans.transforms.append(HidePatch(hide_prob=0.2, grid_sizes=[0, 8]))
    # train_trans.transforms.append(Cutout(length=25))
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_imgs_names, val_imgs_names, test_imgs_names = [], [], []
    for i in train_indices:
        img_name = dataset.imgs[i][0].split('/')[-1].split('_')[0]
        train_imgs_names.append(int(img_name))
    for i in val_indices:
        img_name = dataset.imgs[i][0].split('/')[-1].split('_')[0]
        val_imgs_names.append(int(img_name))
    for i in test_indices:
        img_name = dataset.imgs[i][0].split('/')[-1].split('_')[0]
        test_imgs_names.append(int(img_name))
    print("train indices: ", train_imgs_names)
    print("val indices: ", val_imgs_names)
    print("test indices: ", test_imgs_names)

    if aug_times > 0:
        train_img_list = [dataset.imgs[train_indices[i]][0] for i in range(len(train_indices))]
        train_dset = augmention_dataset(sub_dir=data_dir, class_to_idx=None, image_list=train_img_list,
                                        transform=train_trans)
        train_dset.setmode(2)
        train_dset.maketraindata(aug_times)
        # train_dset.shuffle_data(True)
        train_dataloader = DataLoader(train_dset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=2)
        train_num = len(train_dataloader.dataset)
        train_img_list = [train_dset.samples[i][0] for i in range(train_num)]
    else:
        train_img_list = [dataset.imgs[train_indices[i]][0] for i in range(len(train_indices))]
        train_dset = augmention_dataset(sub_dir=data_dir, class_to_idx=None, image_list=train_img_list,
                                        transform=train_trans)
        train_dset.setmode(1)
        # train_dset.shuffle_data(True)
        train_dataloader = DataLoader(train_dset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=2)
        train_num = len(train_dataloader.dataset)

        # train dataset has no transforms
        # train_sampler = SubsetRandomSampler(train_indices)
        # train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batchsize, drop_last=False)
        # train_num = len(train_dataloader.sampler)

    # val_dataloader = DataLoader(dataset, batch_sampler=valid_sampler)
    # test_dataloader = DataLoader(dataset, batch_sampler=test_sampler)

    val_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=batchsize, pin_memory=True, num_workers=2)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=batchsize, pin_memory=True, num_workers=2)

    val_num, test_num = len(val_dataloader.sampler), len(test_dataloader.sampler)

    return [train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num, train_img_list]


def find_indices_mainview(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_v = []
    top_path, main_path = [], []
    file_names = []
    for i in indices:
        top_path.append(dataset_topview.imgs[i])
        img_name = dataset_topview.imgs[i][0].split('/')[-1].split('_')[0]
        file_names.append(int(img_name))
        img_tuple_0 = (os.path.join(data_dir, "1_mainview", "0_双曲线", img_name + "_v.png"), 0)
        img_tuple_1 = (os.path.join(data_dir, "1_mainview", "1_高亮", img_name + "_v.png"), 1)
        img_tuple_2 = (os.path.join(data_dir, "1_mainview", "2_正常", img_name + "_v.png"), 2)
        if img_tuple_0 in dataset_mainview.imgs:
            img_indice = dataset_mainview.imgs.index(img_tuple_0)
        elif img_tuple_1 in dataset_mainview.imgs:
            img_indice = dataset_mainview.imgs.index(img_tuple_1)
            main_path.append(img_tuple_1)
        else:
            img_indice = dataset_mainview.imgs.index(img_tuple_2)
            main_path.append(img_tuple_2)
        indices_img_v.append(img_indice)
    return indices_img_v, [top_path, main_path], file_names


def find_indices_mainview_ete(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_v = []
    top_path, main_path = [], []
    file_names = []
    for i in indices:
        top_path.append(dataset_topview.imgs[i])
        img_name = dataset_topview.imgs[i][0].split('/')[-1].split('_')[0]
        file_names.append(int(img_name))
        img_tuple_0 = (os.path.join(data_dir, "1_mainview", "0_空洞", img_name + "_v.png"), 0)
        img_tuple_1 = (os.path.join(data_dir, "1_mainview", "1_层间脱空", img_name + "_v.png"), 1)
        img_tuple_2 = (os.path.join(data_dir, "1_mainview", "2_裂缝", img_name + "_v.png"), 2)
        img_tuple_3 = (os.path.join(data_dir, "1_mainview", "3_正常", img_name + "_v.png"), 3)
        if img_tuple_0 in dataset_mainview.imgs:
            img_indice = dataset_mainview.imgs.index(img_tuple_0)
            main_path.append(img_tuple_0)
        elif img_tuple_1 in dataset_mainview.imgs:
            img_indice = dataset_mainview.imgs.index(img_tuple_1)
            main_path.append(img_tuple_1)
        elif img_tuple_2 in dataset_mainview.imgs:
            img_indice = dataset_mainview.imgs.index(img_tuple_2)
            main_path.append(img_tuple_2)
        else:
            img_indice = dataset_mainview.imgs.index(img_tuple_3)
            main_path.append(img_tuple_3)
        indices_img_v.append(img_indice)
    return indices_img_v, [top_path, main_path], file_names


def find_indices_topview(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_h = []
    top_path, main_path = [], []
    file_names = []
    for i in indices:
        main_path.append(dataset_mainview.imgs[i])
        img_name = dataset_mainview.imgs[i][0].split('/')[-1].split('_')[0]
        file_names.append(int(img_name))
        img_tuple_0 = (os.path.join(data_dir, "0_topview", "0_平行线", img_name + "_h.png"), 0)
        img_tuple_1 = (os.path.join(data_dir, "0_topview", "1_暗斑", img_name + "_h.png"), 1)
        img_tuple_2 = (os.path.join(data_dir, "0_topview", "2_亮斑", img_name + "_h.png"), 2)
        img_tuple_3 = (os.path.join(data_dir, "0_topview", "3_正常", img_name + "_h.png"), 3)
        if img_tuple_0 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_0)
            top_path.append(img_tuple_0)
        elif img_tuple_1 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_1)
            top_path.append(img_tuple_1)
        elif img_tuple_2 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_2)
            top_path.append(img_tuple_2)
        else:
            img_indice = dataset_topview.imgs.index(img_tuple_3)
            top_path.append(img_tuple_3)
        indices_img_h.append(img_indice)

    return indices_img_h, [top_path, main_path], file_names


def find_indices_topview_ete(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_h = []
    top_path, main_path = [], []
    file_names = []
    for i in indices:
        main_path.append(dataset_mainview.imgs[i])
        img_name = dataset_mainview.imgs[i][0].split('/')[-1].split('_')[0]
        file_names.append(int(img_name))
        img_tuple_0 = (os.path.join(data_dir, "0_topview", "0_空洞", img_name + "_h.png"), 0)
        img_tuple_1 = (os.path.join(data_dir, "0_topview", "1_层间脱空", img_name + "_h.png"), 1)
        img_tuple_2 = (os.path.join(data_dir, "0_topview", "2_裂缝", img_name + "_h.png"), 2)
        img_tuple_3 = (os.path.join(data_dir, "0_topview", "3_正常", img_name + "_h.png"), 3)
        if img_tuple_0 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_0)
            top_path.append(img_tuple_0)
        elif img_tuple_1 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_1)
            top_path.append(img_tuple_1)
        elif img_tuple_2 in dataset_topview.imgs:
            img_indice = dataset_topview.imgs.index(img_tuple_2)
            top_path.append(img_tuple_2)
        else:
            img_indice = dataset_topview.imgs.index(img_tuple_3)
            top_path.append(img_tuple_3)
        indices_img_h.append(img_indice)

    return indices_img_h, [top_path, main_path], file_names


def find_indices_both_views(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_v, indices_img_h = [], []
    top_path, main_path = [], []
    for i in indices:
        img_name = str(i + 1)
        img_tuple_0_v = (os.path.join(data_dir, "1_mainview", "0_双曲线", img_name + "_v.png"), 0)
        img_tuple_1_v = (os.path.join(data_dir, "1_mainview", "1_高亮", img_name + "_v.png"), 1)
        img_tuple_0_h = (os.path.join(data_dir, "0_topview", "0_平行线", img_name + "_h.png"), 0)
        img_tuple_1_h = (os.path.join(data_dir, "0_topview", "1_暗斑", img_name + "_h.png"), 1)
        img_tuple_2_h = (os.path.join(data_dir, "0_topview", "2_亮斑", img_name + "_h.png"), 2)
        if img_tuple_0_v in dataset_mainview.imgs:
            img_indice_v = dataset_mainview.imgs.index(img_tuple_0_v)
            main_path.append(img_tuple_0_v)
        else:
            img_indice_v = dataset_mainview.imgs.index(img_tuple_1_v)
            main_path.append(img_tuple_1_v)

        if img_tuple_0_h in dataset_topview.imgs:
            img_indice_h = dataset_topview.imgs.index(img_tuple_0_h)
            top_path.append(img_tuple_0_h)
        elif img_tuple_1_h in dataset_topview.imgs:
            img_indice_h = dataset_topview.imgs.index(img_tuple_1_h)
            top_path.append(img_tuple_1_h)
        else:
            img_indice_h = dataset_topview.imgs.index(img_tuple_2_h)
            top_path.append(img_tuple_2_h)

        indices_img_v.append(img_indice_v)
        indices_img_h.append(img_indice_h)

    return [indices_img_h, indices_img_v], [top_path, main_path]


def subset_random_sampler(data_dir, batchsize, seed, ratio=None, aug_times=0, is_upsample=False, view=0):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    dataset_topview = ImageFolder(os.path.join(data_dir, "0_topview"), transform=transformer)
    dataset_mainview = ImageFolder(os.path.join(data_dir, "1_mainview"), transform=transformer)

    # create indices depended on top view classes
    if view == 0:
        train_indices, val_indices, test_indices = get_random_indices(dataset_topview.targets, seed, ratio=ratio,
                                                                      is_upsample=is_upsample)
        # find same file in main view
        train_indices_img, train_indices_path, train_names = find_indices_mainview(data_dir, train_indices,
                                                                                   dataset_topview,
                                                                                   dataset_mainview)
        val_indices_img, val_indices_path, val_names = find_indices_mainview(data_dir, val_indices, dataset_topview,
                                                                             dataset_mainview)
        test_indices_img, test_indices_path, test_names = find_indices_mainview(data_dir, test_indices, dataset_topview,
                                                                                dataset_mainview)

        topview = create_dataloader(dataset_topview, os.path.join(data_dir, "0_topview"), batchsize, train_indices,
                                    val_indices, test_indices, aug_times=aug_times)

        mainview = create_dataloader(dataset_mainview, os.path.join(data_dir, "1_mainview"), batchsize,
                                     train_indices_img, val_indices_img, test_indices_img, aug_times=aug_times)
    else:
        train_indices, val_indices, test_indices = get_random_indices(dataset_mainview.targets, ratio=ratio,
                                                                      is_upsample=is_upsample)
        # find same file in main view
        train_indices_img, train_indices_path, train_names = find_indices_topview(data_dir, train_indices,
                                                                                  dataset_topview,
                                                                                  dataset_mainview, )
        val_indices_img, val_indices_path, val_names = find_indices_topview(data_dir, val_indices, dataset_topview,
                                                                            dataset_mainview)
        test_indices_img, test_indices_path, test_names = find_indices_topview(data_dir, test_indices, dataset_topview,
                                                                               dataset_mainview)

        topview = create_dataloader(dataset_topview, os.path.join(data_dir, "0_topview"), batchsize, train_indices_img,
                                    val_indices_img, test_indices_img, aug_times=aug_times)

        mainview = create_dataloader(dataset_mainview, os.path.join(data_dir, "1_mainview"), batchsize,
                                     train_indices, val_indices, test_indices, aug_times=aug_times)

    # problems_label_tr, problems_label_va, problems_label_te = [], [], []
    # for i in range(3):
    #     for j in range(topview[i + 3]):
    #         if i == 0:
    #             top_type = topview[0].dataset.class_to_idx[topview[0].dataset.samples[j][0].split('/')[-2]]
    #             main_type = mainview[0].dataset.class_to_idx[mainview[0].dataset.samples[j][0].split('/')[-2]]
    #             problems_label_tr.append(get_problem(top_type, main_type))
    #         elif i == 1:
    #             top_type = dataset_topview.targets[val_indices[j]]
    #             main_type = dataset_mainview.targets[val_indices_img[j]]
    #             problems_label_va.append(get_problem(top_type, main_type))
    #         else:
    #             top_type = dataset_topview.targets[test_indices[j]]
    #             main_type = dataset_mainview.targets[test_indices_img[j]]
    #             problems_label_te.append(get_problem(top_type, main_type))

    # 拼接图片并保存
    # combine_img(train_indices_img, train_indices_path, problems_label_tr)
    # combine_img(val_indices_img, val_indices_path, problems_label_va)
    # combine_img(test_indices_img, test_indices_path, problems_label_te)

    # dataset_combineview = ImageFolder(data_dir + "_combine", transform=transformer)
    # combineview = create_dataloader_combine(dataset_combineview, batchsize, train_names, val_names, test_names,
    #                              aug_times=aug_times)

    # return topview, mainview, combineview
    return topview, mainview


def subset_random_sampler_end_to_end(data_dir, batchsize, ratio=None, aug_times=0, is_upsample=False, seed=42, view=0):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset_topview = ImageFolder(os.path.join(data_dir, "0_topview"), transform=transformer)
    dataset_mainview = ImageFolder(os.path.join(data_dir, "1_mainview"), transform=transformer)

    # create indices depended on top view classes
    if view == 0:
        train_indices, val_indices, test_indices = get_random_indices(dataset_topview.targets, seed, ratio=ratio,
                                                                      is_upsample=is_upsample)
        # find same file in main view
        train_indices_img, train_indices_path, train_names = find_indices_mainview_ete(data_dir, train_indices,
                                                                                       dataset_topview,
                                                                                       dataset_mainview)
        val_indices_img, val_indices_path, val_names = find_indices_mainview_ete(data_dir, val_indices, dataset_topview,
                                                                                 dataset_mainview)
        test_indices_img, test_indices_path, test_names = find_indices_mainview_ete(data_dir, test_indices,
                                                                                    dataset_topview,
                                                                                    dataset_mainview)

        topview = create_dataloader(dataset_topview, os.path.join(data_dir, "0_topview"), batchsize, train_indices,
                                    val_indices, test_indices, aug_times=aug_times)
        mainview = create_dataloader(dataset_mainview, os.path.join(data_dir, "1_mainview"), batchsize,
                                     train_indices_img, val_indices_img, test_indices_img, aug_times=aug_times)
    else:
        train_indices, val_indices, test_indices = get_random_indices(dataset_mainview.targets, ratio=ratio,
                                                                      is_upsample=is_upsample)
        # find same file in main view
        train_indices_img, train_indices_path, train_names = find_indices_topview_ete(data_dir, train_indices,
                                                                                      dataset_topview,
                                                                                      dataset_mainview, )
        val_indices_img, val_indices_path, val_names = find_indices_topview_ete(data_dir, val_indices, dataset_topview,
                                                                                dataset_mainview)
        test_indices_img, test_indices_path, test_names = find_indices_topview_ete(data_dir, test_indices,
                                                                                   dataset_topview,
                                                                                   dataset_mainview)

        topview = create_dataloader(dataset_topview, os.path.join(data_dir, "0_topview"), batchsize, train_indices_img,
                                    val_indices_img, test_indices_img, aug_times=aug_times)

        mainview = create_dataloader(dataset_mainview, os.path.join(data_dir, "1_mainview"), batchsize,
                                     train_indices, val_indices, test_indices, aug_times=aug_times)

    # problems_label_tr, problems_label_va, problems_label_te = [], [], []
    # for i in range(3):
    #     for j in range(topview[i + 3]):
    #         if i == 0:
    #             top_type = topview[0].dataset.class_to_idx[topview[0].dataset.samples[j][0].split('/')[-2]]
    #             main_type = mainview[0].dataset.class_to_idx[mainview[0].dataset.samples[j][0].split('/')[-2]]
    #             # top_type = dataset_topview.targets[train_indices[j]]
    #             # main_type = dataset_mainview.targets[train_indices_img[j]]
    #             problems_label_tr.append(get_problem(top_type, main_type))
    #         elif i == 1:
    #             top_type = dataset_topview.targets[val_indices[j]]
    #             main_type = dataset_mainview.targets[val_indices_img[j]]
    #             problems_label_va.append(get_problem(top_type, main_type))
    #         else:
    #             top_type = dataset_topview.targets[test_indices[j]]
    #             main_type = dataset_mainview.targets[test_indices_img[j]]
    #             problems_label_te.append(get_problem(top_type, main_type))
    #
    # remake_img(data_dir, train_indices_img, train_indices_path, problems_label_tr)
    # remake_img(data_dir, val_indices_img, val_indices_path, problems_label_va)
    # remake_img(data_dir, test_indices_img, test_indices_path, problems_label_te)

    return topview, mainview


def remake_img(data_dir, train_indices, train_indices_path, problems_labels):
    problems = ['0_空洞', '1_层间脱空', '2_裂缝', '3_正常']

    for i in range(len(train_indices)):
        top_view_img = Image.open(train_indices_path[0][i][0])
        main_view_img = Image.open(train_indices_path[1][i][0])
        top_save_path = os.path.join(os.path.join(data_dir + "_remake2", "0_topview"), problems[problems_labels[i]])
        main_save_path = os.path.join(os.path.join(data_dir + "_remake2", "1_mainview"), problems[problems_labels[i]])
        if not os.path.exists(top_save_path):
            os.makedirs(top_save_path)
        if not os.path.exists(main_save_path):
            os.makedirs(main_save_path)
        top_view_img.save(
            os.path.join(top_save_path, train_indices_path[0][i][0].split('/')[-1].split('_')[0] + "_h.png"))
        main_view_img.save(
            os.path.join(main_save_path, train_indices_path[0][i][0].split('/')[-1].split('_')[0] + "_v.png"))


def combine_img(train_indices, train_indices_path, problems_labels):
    problems = ['0_空洞', '1_层间脱空', '2_裂缝']

    for i in range(len(train_indices)):
        top_view_img = Image.open(train_indices_path[0][i][0])
        main_view_img = Image.open(train_indices_path[1][i][0])
        img_combine = Image.new('RGB', (top_view_img.width, top_view_img.height + main_view_img.height))
        img_combine.paste(top_view_img, (0, 0, top_view_img.width, top_view_img.height))
        img_combine.paste(main_view_img,
                          (0, top_view_img.height, top_view_img.width, main_view_img.height + top_view_img.height))
        save_path = os.path.join(data_dir + "_combine", problems[problems_labels[i]])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_combine.save(os.path.join(save_path, train_indices_path[0][i][0].split('/')[-1].split('_')[0] + ".png"))


def subset_random_sampler_combine(data_dir, batchsize, ratio=None, aug_times=0, is_upsample=False):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset_combineview = ImageFolder(data_dir + "_combine", transform=transformer)
    # create indices
    train_indices, val_indices, test_indices = get_random_indices(dataset_combineview.targets, ratio=ratio,
                                                                  is_upsample=is_upsample)

    combineview = create_dataloader(dataset_combineview, data_dir + "_combine", batchsize, train_indices, val_indices,
                                    test_indices, aug_times=aug_times)

    return combineview


def get_problem(top_view_type, main_view_type):
    problems = ['空洞', '层间脱空', '裂缝', '正常']
    if main_view_type == 1:
        return 0
    else:
        if top_view_type == 1 or top_view_type == 2:
            return 1
        elif top_view_type == 0:
            return 2
        else:
            return 3

# if __name__ == '__main__':
#     data_dir = './radar_data'
#     # train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num = random_split(data_dir, batchsize,
#     #                                                                                                ratio=[0.6, 0.2, 0.2],
#     #                                                                                                aug_times=5)
#     # top_list, main_list, combinelist = subset_random_sampler_combine(data_dir, 32, ratio=[0.5, 0.25, 0.25], aug_times=0)
#     # print(temp)
#     top_list, main_list = subset_random_sampler(data_dir, batchsize=32, ratio=[0.5, 0.25, 0.25], aug_times=0,
#                                                 view=1)
#     print(top_list[3], top_list[4], top_list[5])
#     print(main_list[3], main_list[4], main_list[5])
#     a, b = [], []
#     for i in range(top_list[3]):
#         a.append(top_list[0].dataset.samples[i][0].split("/")[-2])
#         b.append(main_list[0].dataset.samples[i][0].split("/")[-2])
#
#     print(a)
#     print(b)
#     #
#     labels_all_t_tr = np.array([], dtype=int)
#     labels_all_m_tr = np.array([], dtype=int)
#     img_all_t_tr = np.array([], dtype=int)
#     img_all_m_tr = np.array([], dtype=int)
#     # data_main = iter(main_list[0])
#     # for i, data in enumerate(top_list[0], 1):
#     #     images, labels = data
#     #     (images_main, labels_main) = next(data_main)
#     #
#     #     labels_all_t_tr = np.append(labels_all_t_tr, labels)
#     #     labels_all_m_tr = np.append(labels_all_m_tr, labels_main)
#     #     img_all_t_tr = np.append(img_all_t_tr, images)
#     #     img_all_m_tr = np.append(img_all_m_tr, images_main)
#     #
#     # print(labels_all_m_tr)
#     import matplotlib.pyplot as plt
#
#
#     def imshow(img):
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.savefig("example.pdf", dpi=400)
#         plt.show()
#
#
#     data_main = iter(main_list[0])
#     (images_main, labels_main) = next(data_main)
#     data_top = iter(top_list[0])
#     (images_top, labels_top) = next(data_top) #1 5 0
#     img_m = np.empty([7,3,224,224])
#     img_m[0,:]=images_top[1]
#     img_m[1,:]=images_top[13]
#     img_m[2,:]=images_top[5]
#     img_m[3, :] = images_top[0]
#     img_m[4, :] = images_main[1]
#     img_m[5, :] = images_main[5]
#     img_m[6, :] = images_main[0]
#     img_torch = torch.from_numpy(img_m)
#     imshow(torchvision.utils.make_grid(img_torch))
#     print(images_top[1], images_top[13], images_top[5], images_top[0])
#
#         # problems_label = np.array([], dtype=int)
#         # for idx in range(labels.shape[0]):
#         #     problems_label = np.append(problems_label, get_problem(labels[idx], labels_main[idx]))
#         # problems_label_torch = torch.from_numpy(problems_label)
#     #
#     # for step, (x2, y2) in enumerate(top_list[0], 1):
#     #     # print(x2.shape)
#     #     # print(y2)
#     #     labels_all_t_tr = np.append(labels_all_t_tr, y2)
#     #
#     # for step, (x2, y2) in enumerate(main_list[0], 1):
#     #     # print(x2.shape)
#     #     # print(y2)
#     #     labels_all_m_tr = np.append(labels_all_m_tr, y2)
#     # print(labels_all_t_tr)
#     # print(labels_all_m_tr)
