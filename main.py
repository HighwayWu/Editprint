import os
import cv2
import rawpy
import shutil
import random
import datetime
import argparse
import importlib
import numpy as np
import logging as logger
import albumentations as A
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, rep=1, isTest=False):
        self.data_root = data_root
        self.data_size = 512
        self.data_size_test = 1536
        self.rep = rep
        self.filelist = []
        self.filedict = {}
        self.isTest = isTest
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, text_label, num_label = line.split('$@')
            num_label = int(num_label)

            self.filelist.append((image_path, text_label.replace('_', ' '), num_label))

            if num_label in self.filedict.keys():
                value = self.filedict[num_label]
                value.append(image_path)
            else:
                value = [image_path]
            self.filedict.update({num_label: value})

            line = train_file_buf.readline().strip()

        self.albu_pre = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size_test, min_width=self.data_size_test, p=1.0),
            A.CenterCrop(height=self.data_size_test, width=self.data_size_test, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if not self.isTest:
            self.generate_aug_list(num=9999)
            self.generate_raw_aug_list()

    def generate_aug_list(self, num=0):
        aug_candidate_list = ['resize', 'compression', 'blur', 'noise']
        resize_factor_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        resize_factor_name_list = ['nearest', 'linear', 'cubic', 'area']
        quality_factor_list = list(range(50, 98))
        blur_kernel_list = [3, 5, 7, 9, 11, 13]
        noise_variance_list = list(range(3, 10))

        self.albu_aug_list = []
        self.albu_aug_list_text = []
        if num == 0:
            self.albu_aug_list = [A.PadIfNeeded(min_height=512, min_width=512, p=1.0)]
            self.albu_aug_list_text = ['Identity']
            return
        for _ in range(num):
            cur_aug = []
            cur_text = ''
            for _ in range(np.random.randint(1, 4)):
                candid = random.sample(aug_candidate_list, 1)[0]
                if candid == 'resize':
                    param_idx = random.sample(list(range(len(resize_factor_list))), 1)[0]
                    param_resize = resize_factor_list[param_idx]
                    param_name = resize_factor_name_list[param_idx]
                    param_scale = random.sample([-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5], 1)[0]
                    cur_aug.append(A.Compose([A.RandomScale(scale_limit=(param_scale, param_scale), interpolation=param_resize, p=1.0),
                                              A.Resize(self.data_size, self.data_size, interpolation=param_resize, p=1.0)]))
                    cur_text += 'Resize-%s ' % (param_name)
                if candid == 'compression':
                    param_jpeg = random.sample(quality_factor_list, 1)[0]
                    param_type = random.sample([0, 1], 1)[0]
                    cur_aug.append(A.ImageCompression(quality_lower=param_jpeg, quality_upper=param_jpeg, compression_type=param_type, p=1.0))
                    param_type = 'JPEG' if param_type == 0 else 'WEBP'
                    cur_text += '%s-%d ' % (param_type, param_jpeg)
                if candid == 'blur':
                    param_blur = random.sample(blur_kernel_list, 1)[0]
                    param_type = random.sample(['Blur', 'GaussianBlur', 'MedianBlur', 'MotionBlur'], 1)[0]
                    if param_type == 'Blur':
                        cur_aug.append(A.Blur(blur_limit=(param_blur, param_blur), p=1.0))
                    elif param_type == 'GaussianBlur':
                        cur_aug.append(A.GaussianBlur(blur_limit=(param_blur, param_blur), p=1.0))
                    elif param_type == 'MedianBlur':
                        cur_aug.append(A.MedianBlur(blur_limit=(param_blur, param_blur), p=1.0))
                    elif param_type == 'MotionBlur':
                        cur_aug.append(A.MotionBlur(blur_limit=(param_blur, param_blur), p=1.0))
                    cur_text += '%s-%d ' % (param_type, param_blur)
                if candid == 'noise':
                    param_type = random.sample(['GaussNoise', 'ISONoise'], 1)[0]
                    if param_type == 'GaussNoise':
                        param_noise = random.sample(noise_variance_list, 1)[0]
                        cur_aug.append(A.GaussNoise(var_limit=(param_noise, param_noise), p=1.0))
                    elif param_type == 'ISONoise':
                        param_noise = random.sample([0.1, 0.2, 0.3, 0.4, 0.5], 1)[0]
                        cur_aug.append(A.ISONoise(intensity=(param_noise, param_noise), p=1.0))
                    cur_text += '%s-%s ' % (param_type, str(param_noise))
            cur_aug = A.Compose(cur_aug, p=1.0)

            self.albu_aug_list.append(cur_aug)
            self.albu_aug_list_text.append(cur_text)

    def generate_raw_aug_list(self):
        demosaic_list = {
            'DM-AHD': rawpy.DemosaicAlgorithm.AHD,
            'DM-DCB': rawpy.DemosaicAlgorithm.DCB,
            'DM-DHT': rawpy.DemosaicAlgorithm.DHT,
            'DM-PPG': rawpy.DemosaicAlgorithm.PPG,
        }
        wb_list = {
            'WB-Camera': [True, False, None],
            'WB-Auto': [False, True, None],
            'WB-User': [False, False, [1, 1, 1, 1]],
        }
        gamma_list = {
            # 'Gamma-None': (1, 1),
            'Gamma-Default': (2.222, 4.5),
        }
        tone_list = {
            'Tone-Scale-Bright': [False, False],
            'Tone-Scale': [False, True],
            'Tone-Bright': [True, False],
        }
        key_list, value_list = [], []
        for key1 in demosaic_list:
            for key2 in wb_list:
                for key3 in tone_list:
                    key_list.append(key1 + ' ' + key2 + ' ' + key3 + ' ')
                    value_list.append(
                        [demosaic_list[key1], wb_list[key2], tone_list[key3]]
                    )
        self.raw_text_list = key_list
        self.raw_value_list = value_list

    def transform(self, x):
        if self.isTest:
            x = self.albu_pre_val(image=x)['image']
        else:
            x = self.albu_pre(image=x)['image']
        return x

    def aug_transform(self, x, rnd_aug_index=-1):
        if rnd_aug_index >= 0:
            x = self.albu_aug_list[rnd_aug_index](image=x)['image']
            x = x.astype(np.uint8)
        return x

    def __len__(self):
        if self.isTest:
            return len(self.filelist)
        else:
            return 99999

    def __getitem__(self, index):
        if self.isTest:
            return self.getitem_test(index, self.filelist)
        else:
            return self.getitem_raw()

    def getitem_test(self, index, data_list):
        image_path, text_label, image_label = data_list[index]

        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, image_path)

        image = cv2.imread(image_path)[..., ::-1]
        crop_list, label_list = None, []
        crop = self.transform(image)
        crop = self.imagenet_norm(crop).unsqueeze(0)
        crop_list = torch.cat([crop_list, crop]) if crop_list is not None else crop
        label_list.append(image_label)
        label_list = torch.LongTensor(label_list)
        return crop_list, label_list

    # Load RAW image
    def getitem_raw(self):
        image_label = random.sample(self.filedict.keys(), 1)[0]
        image_path_list = random.sample(self.filedict[image_label], self.rep)

        raw_idx = random.sample(range(len(self.raw_text_list)), 1)[0]
        raw_text = self.raw_text_list[raw_idx]
        raw_value = self.raw_value_list[raw_idx]
        image_list = []
        for image_path in image_path_list:
            if not os.path.exists(image_path):
                image_path = os.path.join(self.data_root, image_path)

            with rawpy.imread(image_path) as raw:
                img = raw.postprocess(
                    half_size=True,
                    demosaic_algorithm=raw_value[0],
                    use_camera_wb=raw_value[1][0],
                    use_auto_wb=raw_value[1][1],
                    user_wb=raw_value[1][2],
                    # gamma=raw_value[2],
                    no_auto_scale=raw_value[2][0],
                    no_auto_bright=raw_value[2][1],
                    output_bps=8
                )
            image_list.append(img)

        crop_list, num_label_list, text_label_list = None, [], []
        aug_num = args.batch_aug_num
        aug_ind_list = random.sample(range(len(self.albu_aug_list)), aug_num)
        for image in image_list:
            # image = image[..., ::-1]
            crop_backup = self.transform(image)
            for inner_idx in range(aug_num):
                aug_ind = aug_ind_list[inner_idx]
                crop = self.aug_transform(crop_backup, aug_ind)
                crop = self.imagenet_norm(crop).unsqueeze(0)
                crop_list = torch.cat([crop_list, crop]) if crop_list is not None else crop
                num_label_list.append(image_label)
                text_label_list.append(raw_text + self.albu_aug_list_text[aug_ind])
        num_label_list = torch.LongTensor(num_label_list)
        return crop_list, num_label_list, text_label_list

    # Load RGB image
    def getitem_rgb(self):
        image_label = random.sample(self.filedict.keys(), 1)[0]
        image_path_list_init = random.sample(self.filedict[image_label], self.rep)

        image_path_list = []
        for tmp in image_path_list_init:
            if not os.path.exists(tmp):
                tmp = os.path.join(self.data_root, tmp)
            image_path_list.append(tmp)

        image_list = [cv2.imread(tmp)[..., ::-1] for tmp in image_path_list]
        crop_list, num_label_list, text_label_list = None, [], []
        aug_num = args.batch_aug_num
        aug_ind_list = random.sample(range(len(self.albu_aug_list)), aug_num)
        for image in image_list:
            crop_backup = self.transform(image)
            for inner_idx in range(aug_num):
                aug_ind = aug_ind_list[inner_idx]
                crop = self.aug_transform(crop_backup, aug_ind)
                crop = self.imagenet_norm(crop).unsqueeze(0)
                crop_list = torch.cat([crop_list, crop]) if crop_list is not None else crop
                num_label_list.append(image_label)
                text_label_list.append(self.albu_aug_list_text[aug_ind])
        num_label_list = torch.LongTensor(num_label_list)
        return crop_list, num_label_list, text_label_list


def train_one_epoch(data_loader, model, model_lp, optimizer, optimizer_fc, cur_epoch, args):
    loss_meter = AverageMeter()
    loss_fc_meter = AverageMeter()
    loss_meter.reset()
    loss_fc_meter.reset()
    batch_idx = 0
    for (images, _, dup_text_labels) in data_loader:
        batch_num, rep, C, H, W = images.shape
        images = images.cuda().reshape(batch_num * rep, C, H, W)

        # Remove duplicated text labels, and create numerical labels
        text_labels = []
        num_labels = []
        for iter_1 in range(len(dup_text_labels[0])):
            for iter_2 in range(len(dup_text_labels)):
                cur_tmp_label = dup_text_labels[iter_2][iter_1]
                if cur_tmp_label not in text_labels:
                    text_labels.append(cur_tmp_label)
                num_labels.append(torch.tensor(text_labels.index(cur_tmp_label)))
        num_labels = torch.stack(num_labels).cuda().flatten().squeeze()

        features_logits, image_feats, text_feats, cos_sim = model(images, text_labels, isTrain=True)

        cluster_feats = model.module.clusterU(image_feats, text_feats, num_labels)

        model_lp.module.fc_init()
        for _ in range(10):
            prob = model_lp(image_feats.detach())
            loss_tmp = args.criterion_ce(prob, num_labels)
            optimizer_fc.zero_grad()
            loss_tmp.backward()
            optimizer_fc.step()

        prob = model_lp(image_feats)

        new_logits = model.module.get_scale() * cluster_feats @ text_feats.t()
        loss_image = args.criterion_ce(new_logits, torch.arange(new_logits.shape[0]).to(new_logits.device))
        text_logits = new_logits.t()
        loss_text = args.criterion_ce(text_logits, torch.arange(text_logits.shape[0]).to(text_logits.device))
        loss_clip = (loss_image + loss_text) / 2

        soft_labels = []
        template_matrix = cos_sim.detach().cpu().numpy()
        labels = num_labels
        for label in labels:
            cur_temp_label = template_matrix[label]
            cur_temp_label[cur_temp_label < 0] = 0
            cur_temp_label[label] = np.sum(cur_temp_label)
            cur_temp_label = torch.tensor(cur_temp_label)
            cur_temp_label[cur_temp_label != 0] = cur_temp_label[cur_temp_label != 0].softmax(dim=0)

            soft_labels.append(cur_temp_label)
        soft_labels = torch.stack(soft_labels)
        soft_labels = F.pad(soft_labels, (0, batch_num * args.batch_aug_num - soft_labels.shape[1], 0, 0))
        loss_fc = args.criterion_ce(prob, soft_labels.to(prob.device))

        loss = (loss_clip + loss_fc)

        optimizer.zero_grad()
        optimizer_fc.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer_fc.step()

        loss_meter.update(loss_clip.item(), batch_num*rep)
        loss_fc_meter.update(loss_fc.item(), batch_num*rep)
        if batch_idx % 10 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Ep %03d, it %03d/%03d, lr: %8.7f, CA: %7.6f, CL: %7.6f' % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg, loss_fc_meter.avg))
            loss_meter.reset()
            loss_fc_meter.reset()
        if batch_idx > 50:
            break
        batch_idx += 1

    return loss_avg


def testing_open_verification(model, txt_file, args):
    data_loader = DataLoader(
        ImageDataset(args.data_root, txt_file, isTest=True),
        args.batch_size_test,
        num_workers=min(48, args.batch_size_test),
        shuffle=False,
    )
    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    label2features_dict = {}
    for (images, labels) in data_loader:
        images = images.cuda()
        b, rep, C, H, W = images.shape
        images = images.reshape(b * rep, C, H, W).contiguous()
        labels = labels.flatten().cpu().numpy()

        with torch.no_grad():
            feats = model(image_input=images, isTrain=False)
            feats = F.normalize(feats, dim=1)
            for feat, label in zip(feats, labels):
                if label in label2features_dict.keys():
                    features = label2features_dict[label]
                    features.append(feat)
                else:
                    features = [feat]
                label2features_dict.update({label: features})

    keys = label2features_dict.keys()
    for _ in range(10000):
        pos = random.sample([0, 1], 1)[0]
        if pos == 1:
            rnd_label = random.sample(keys, 1)[0]
            feat1, feat2 = random.sample(label2features_dict[rnd_label], 2)
        else:
            label1, label2 = random.sample(keys, 2)
            feat1 = random.sample(label2features_dict[label1], 1)[0]
            feat2 = random.sample(label2features_dict[label2], 1)[0]
        cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
        gt_labels_list.append(pos)
        prob_labels_list.append(cos_sim[0].cpu().numpy())

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    return auc


def testing_close_classification(model, txt_file, args):
    data_loader = DataLoader(
        ImageDataset(args.data_root, txt_file, isTest=True),
        args.batch_size_test,
        num_workers=min(48, args.batch_size_test),
        shuffle=False,
    )
    label2features_dict = {}
    for (images, labels) in data_loader:
        images = images.cuda()
        b, rep, C, H, W = images.shape
        images = images.reshape(b * rep, C, H, W).contiguous()
        labels = labels.flatten().squeeze().cpu().numpy()

        with torch.no_grad():
            feats = model(image_input=images, isTrain=False)
            feats = F.normalize(feats, dim=1)

            for feat, label in zip(feats, labels):
                if label in label2features_dict.keys():
                    features = label2features_dict[label]
                    features.append(feat)
                else:
                    features = [feat]
                label2features_dict.update({label: features})

    label2candidate_features_dict = {}
    label2anchor_features_dict = {}
    keys = label2features_dict.keys()
    keys = sorted(keys)

    anchor_num = 2
    for key in keys:
        value = label2features_dict[key]
        anchor = torch.mean(torch.stack(value[:anchor_num]), dim=0)
        label2anchor_features_dict.update({key: anchor})
        label2candidate_features_dict.update({key: value[anchor_num:]})

    confusion_mat = np.zeros((len(keys), len(keys)))
    for idx, candid_key in enumerate(keys):
        candid_list = label2candidate_features_dict[candid_key]
        for feat in candid_list:
            max_sim, max_idx = -1, -1
            for y_idx, y_key in enumerate(keys):
                anchor = label2anchor_features_dict[y_key]
                feat1, feat2 = feat, anchor
                cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    max_idx = y_idx
            confusion_mat[idx][max_idx] += 1

    # logger.info(confusion_mat)

    conf_precision_list, conf_recall_list = [], []
    for row in range(len(confusion_mat)):
        data_list = confusion_mat[row, :]
        if np.sum(data_list) == 0:
            conf_recall_list.append(None)
            continue
        recall = data_list[row] / np.sum(data_list)
        conf_recall_list.append(recall)
    for col in range(len(confusion_mat)):
        data_list = confusion_mat[:, col]
        if np.sum(data_list) == 0:
            conf_precision_list.append(None)
            continue
        precision = data_list[col] / np.sum(data_list)
        conf_precision_list.append(precision)
    avg_f1 = []
    for p1, r1 in zip(conf_precision_list, conf_recall_list):
        if p1 is None or r1 is None:
            continue
        if p1 + r1 == 0:
            avg_f1.append(0)
        else:
            avg_f1.append(2 * (p1 * r1) / (p1 + r1))
    avg_f1 = np.mean(avg_f1)
    avg_recall = np.mean(list(filter(None, conf_recall_list)))
    avg_precision = np.mean(list(filter(None, conf_precision_list)))

    return avg_precision, avg_recall, avg_f1


def model_init(isTrain=True, resume=''):
    model = getattr(importlib.import_module('model'), args.model)()
    model = torch.nn.DataParallel(model).cuda()

    model_lp = getattr(importlib.import_module('model'), 'LinearProbe')(in_channel=1024, out_channel=args.batch_size * args.batch_aug_num)
    model_lp = torch.nn.DataParallel(model_lp).cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model Params: %.2f' % (params / (1024 ** 2)))

    if resume != '':
        pre_dict = torch.load(resume)
        cur_dict = model.state_dict()
        for pre_key in pre_dict.keys():
            if pre_key in cur_dict.keys() and pre_dict[pre_key].shape == cur_dict[pre_key].shape:
                cur_dict.update({pre_key: pre_dict[pre_key]})
            else:
                logger.info('Error in loading %s' % pre_key)
        model.load_state_dict(cur_dict)
        logger.info('Loaded [%s].' % resume)

    if isTrain:
        return model, model_lp
    else:
        return model


def train(args):
    data_loader = DataLoader(
        ImageDataset(args.data_root, 'data/fivek_img10_cat1.txt', rep=args.batch_rep_num),
        args.batch_size,
        shuffle=True,
        num_workers=min(48, args.batch_size))
    data_loader.dataset.generate_aug_list(9999)

    model, model_lp = model_init(isTrain=True, resume=args.resume)
    # model, model_lp = model_init(isTrain=True, resume='')
    model.train()
    model_lp.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(parameters, lr=args.lr)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    optimizer_fc = optim.AdamW([p for p in model_lp.parameters() if p.requires_grad], lr=args.lr)
    lr_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    for epoch in range(999):
        train_one_epoch(data_loader, model, model_lp, optimizer, optimizer_fc, epoch, args)

        torch.save(model.state_dict(), os.path.join(args.out_dir, 'lasted_model.pt'))
        lr_schedule.step()


def test(args):
    test_file_list = [
        ('data/FODB_img60_cat6.txt', 'FODB_6osn'),
    ]

    model = model_init(isTrain=False, resume=args.resume)
    model.eval()

    for txt_file, nickname in test_file_list:
        # Open Verification:
        auc = testing_open_verification(model, txt_file, args)
        logger.info('%s in Open Verification: AUC %5.4f' % (nickname, auc))

        # Close Classification:
        prc, rcl, f1 = testing_close_classification(model, txt_file, args)
        logger.info('%s in Close Classification: PRC %5.4f, RCL %5.4f, F1 %5.4f' % (nickname, prc, rcl ,f1))
    return


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument('--train', action='store_true', default=False)
    conf.add_argument('--test', action='store_true', default=False)
    conf.add_argument('--model', type=str, default='EditprintFramework')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument('--out_dir', type=str, default='out_dir', help="The folder to save models.")
    conf.add_argument('--batch_size', type=int, default=8, help='Number of unique image')
    conf.add_argument('--batch_aug_num', type=int, default=2, help='Number of augment chains')
    conf.add_argument('--batch_rep_num', type=int, default=4, help='Number of repeated construction')
    conf.add_argument('--batch_size_test', type=int, default=16)
    conf.add_argument('--data_root', type=str, default='data/')
    conf.add_argument('--data_size', type=int, default=256, help='The image size for training.')
    conf.add_argument('--data_size_test', type=int, default=1536, help='The image size for testing.')
    conf.add_argument('--resume', type=str, default='weights/editprint_model.pt')
    conf.add_argument('--gpu', type=str, default='0', help='The gpu')
    args = conf.parse_args()
    args.criterion_ce = torch.nn.CrossEntropyLoss().cuda()

    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.info(args)
    args.isTrain = False if args.test else True
    if args.isTrain:
        date_now = datetime.datetime.now()
        date_now = '/Log_v%02d%02d%02d%02d' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
        args.time = date_now
        args.out_dir = args.out_dir + args.time
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)

        train(args)
    else:
        test(args)
