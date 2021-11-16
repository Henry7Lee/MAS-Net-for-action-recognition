import os
import torch
import torchvision
import torchvision.datasets as datasets


#ROOT_DATASET = './'


def return_ucf101(ROOT_DATASET):
    filename_categories = 'UCF101/labels/classInd.txt'
    root_data = ROOT_DATASET + 'UCF101/jpg'
    filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
    filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
    prefix = 'img_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_hmdb51(ROOT_DATASET):
    filename_categories = 51
    root_data = ROOT_DATASET + 'HMDB51/images'
    filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
    filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
    prefix = 'img_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(ROOT_DATASET):
    filename_categories = 'jester/category.txt'
    prefix = '{:05d}.jpg'
    root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
    filename_imglist_train = 'jester/train_videofolder.txt'
    filename_imglist_val = 'jester/val_videofolder.txt'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_diving48(ROOT_DATASET):
    #filename_categories = 48
    root_data = ROOT_DATASET + 'Diving48/frames'
    filename_imglist_train = 'Diving48/train.txt'
    filename_imglist_val = 'Diving48/valid.txt'
    prefix = 'frames{:05d}.jpg'
    return filename_categories,filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics(ROOT_DATASET):
    filename_categories = 400
    root_data = ROOT_DATASET + 'K400/img'
    filename_imglist_train = 'K400/train_videofolder.txt'
    filename_imglist_val = 'K400/val_videofolder.txt'
    prefix = 'img_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv1(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = ROOT_DATASET + 'somethingv1/20bn-something-something-v1'
    filename_imglist_train = 'somethingv1/train.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv2(ROOT_DATASET):
    filename_categories = 'somethingv2/category.txt'
    root_data = ROOT_DATASET + 'somethingv2/20bn-something-something-v2-frames'
    filename_imglist_train = 'somethingv2/train.txt'
    filename_imglist_val = 'somethingv2/valid.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv1_mini(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = ROOT_DATASET + 'somethingv1/20bn-something-something-v1'
    filename_imglist_train = 'somethingv1/trainQ.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset,ROOT_DATASET):
    dict_single = {'jester': return_jester, 'somethingv1': return_somethingv1, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,'diving48': return_diving48,
                   'kinetics': return_kinetics , 'somethingv1_mini': return_somethingv1_mini }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](ROOT_DATASET)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)

    #file_categories = os.path.join(ROOT_DATASET, file_categories)

    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories

    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))

    return categories, file_imglist_train, file_imglist_val, root_data, prefix

