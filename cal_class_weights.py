import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def is_label_BDD(filename):
    return filename.endswith("_train_id.png")


def is_label_IDD(filename):
    return filename.endswith("_labellevel3Ids.png")


def is_label_city(filename):
    return filename.endswith("_labelTrainIds.png")


def calc_weights(args, enc=False):
    datapath = args.datadir  # /bdd100k/seg/ for bdd
    dataset = args.dataset
    num_classes = args.num_classes
    print('inside calc_weights\n')
    print('datapath: ', datapath)
    print('dataset: ', dataset)
    print('classes: ', num_classes)

    if dataset == 'cityscapes' or dataset == 'IDD':
        datapath = os.path.join(datapath, 'gtFine/train/')
    elif dataset == 'BDD':
        datapath = os.path.join(datapath, 'labels/train/')
    print('calculating weights for {} with {} classes, located in root dir: {}'.format(
        dataset, num_classes, datapath))

    if dataset == 'IDD':
        label_file_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(datapath)) for f in fn if is_label_IDD(f)]
    elif dataset == 'cityscapes':
        label_file_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(datapath)) for f in fn if is_label_city(f)]
    elif dataset == 'BDD':
        label_file_list = os.listdir(datapath)
        label_file_list = [os.path.join(datapath, fn) for fn in label_file_list]
#    print(label_file_list)

    gt_pix_count = np.zeros(num_classes)
    for file in label_file_list:
        label = cv2.imread(file, 0)
        if label is not None:
            label_id, pix_c = np.unique(label, return_counts=True)
            for i, j in zip(label_id, pix_c):
                if i == 255:
                    gt_pix_count[num_classes-1] += j
                else:
                    gt_pix_count[i] += j
    gt_pix_count += 1
    class_prob = gt_pix_count / np.sum(gt_pix_count)
    if not (enc):
        class_prob += 1.1  # not enc
        print('hi inside p=1.1')
    else:
        class_prob += 1.2  # enc
        print('hi inside p=1.2')
    weight = np.reciprocal(np.log(class_prob))
    print('making the last value {} to 0'.format(weight[num_classes-1]))
    weight[num_classes-1] = 0

    return weight
