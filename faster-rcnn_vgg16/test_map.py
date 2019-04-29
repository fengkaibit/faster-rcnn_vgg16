from __future__ import absolute_import
import cupy as cp
import os
import matplotlib
import ipdb
import torch
from tqdm import tqdm  #进度条库

from utils.config import opt
from data.dataset import TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.eval_tool import eval_detection_voc

import resource  #用于查询或修改当前系统资源限制设置

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_ , pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def test(**kwargs):
    opt.load_path = \
        '/home/fengkai/PycharmProjects/my-faster-rcnn/checkpoints/fasterrcnn_04231732_0.6941460588341642'
    opt._parse(kwargs)  #将调用函数时候附加的参数用，config.py文件里面的opt._parse()进行解释，然后获取其数据存储的路径，之后放到Dataset里面！

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       #pin_memory=True
                                       )   #pin_memory锁页内存,开启时使用显卡的内存，速度更快


    faster_rcnn = FasterRCNNVGG16()
    print(faster_rcnn)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #判断opt.load_path是否存在，如果存在，直接从opt.load_path读取预训练模型，然后将训练数据的label进行可视化操作


    trainer.load(opt.load_path)
    print('load pretrained model from %s' %opt.load_path)

    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
    print('map is: ', str(eval_result['map']))

if __name__ == '__main__':
    test()
