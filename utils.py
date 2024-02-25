import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import lmdb
import pickle
import six

import os
from PIL import Image

class Normalize(nn.Module):
    def __init__(self, ms=None):
        super(Normalize, self).__init__()
        if ms == None:
            self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.ms[0][i]) / self.ms[1][i]
        return x

class CUB200_PIL(Dataset):
    def __init__(self, data_path, img_filepath):
        self.img_path = data_path
        # reading img file from file
        fp = open(img_filepath, 'rt')
        self.img_filename = []
        self.label = []
        for x in fp:
            x = x.strip()
            self.img_filename.append(x.split(' ')[0])
            self.label.append(int(x.strip().split(' ')[1]))
        fp.close()

    def __getitem__(self, index):
        with open(os.path.join(self.img_path, self.img_filename[index]), 'rb') as f:
            bin_data = f.read()
        label = self.label[index]
        return bin_data, label

    def __len__(self):
        return len(self.img_filename)

class CUB200_PIL_nolabel(Dataset):

    def __init__(self, data_path, img_filepath, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filepath, 'rt')
        self.img_filename = []
        for x in fp:
            x = x.strip()
            self.img_filename.append(x)
        fp.close()


    def __getitem__(self, index):
        f = open(os.path.join(self.img_path, self.img_filename[index]), 'rb')
        img = Image.open(f)
        img = img.convert('RGB')
        imgsize = img.size
        if self.transform is not None:
            img = self.transform(img)
        return img, (imgsize[1], imgsize[0])#, label#, self.img_filename[index]

    def __len__(self):
        return len(self.img_filename)


class ElementwisePair(CUB200_PIL_nolabel):

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, self.img_filename[index]

class ImageFolderPair(torchvision.datasets.folder.ImageFolder):
    """Pair Dataset.
    """

    def __getitem__(self, index):
        img = Image.open(self.imgs[index][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, self.imgs[index][1]


class ImageFolderTriple(torchvision.datasets.folder.ImageFolder):
    """triplet Dataset.
    """
    def __init__(self, root, transform, trans_ori):
        super().__init__(root, transform)
        self.trans_ori = trans_ori

    def __getitem__(self, index):
        img = Image.open(self.imgs[index][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        pos = self.trans_ori(img)

        return pos_1, pos_2, pos, self.imgs[index][1]


class ElementwiseTriple(CUB200_PIL_nolabel):

    def __init__(self, data_path, img_filepath, transform, trans_ori):
        super().__init__(data_path, img_filepath, transform)
        self.trans_ori = trans_ori

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        pos = self.trans_ori(img)
        return pos_1, pos_2, pos, self.img_filename[index]


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, trans_ori=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = self.loads_data(txn.get(b'__len__'))
            self.keys = self.loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.trans_ori = trans_ori

    def loads_data(self, buf):
        """
        Args:
            buf: the output of `dumps`.
        """
        return pickle.loads(buf)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = self.loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        # target = unpacked[1]

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        pos = self.trans_ori(img)

        return pos_1, pos_2, pos


    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
