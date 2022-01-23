import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label_city(filename):
    return filename.endswith("_labelTrainIds.png")


def is_label_IDD(filename):
    return filename.endswith("_labellevel3Ids.png")


def is_label_BDD(filename):
    return filename.endswith("_train_id.png")


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


def image_path_city(root, name):
    return os.path.join(root, f'{name}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
                          for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        print(self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        # [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        # self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.labels_root)) for f in fn if is_label_city(f)]
        self.filenamesGt.sort()
        # self.filenames = self.filenames[:1]
        # self.filenamesGt = self.filenamesGt[:1]  # trying to plot the t-sne

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)

# added


class IDD(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        print(self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.labels_root)) for f in fn if is_label_IDD(f)]
        self.filenamesGt.sort()

#        self.filenames = self.filenames[:20]
#        self.filenamesGt = self.filenamesGt[:20]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        # image_path_city will work for IDD also as the images already have a .png extension
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)


class BDD(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'images/')
        self.labels_root = os.path.join(root, 'labels/')

        self.images_root += subset
        self.labels_root += subset

        print(self.images_root)
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [f for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        # [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        # self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [fn for fn in os.listdir(self.labels_root) if is_label_BDD(fn)]
        self.filenamesGt.sort()

#        self.filenames = self.filenames[:20]
#        self.filenamesGt = self.filenamesGt[:20]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)
