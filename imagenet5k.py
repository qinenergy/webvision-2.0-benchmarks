#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Imagenet5k.py

import os
import tarfile
import numpy as np
import tqdm

from tensorpack.utils import logger
from tensorpack.utils.loadcaffe import get_caffe_pb
#from tensorpack.utils.fs import mkdir_p, download, get_dataset_path
from tensorpack.utils.timer import timed_operation
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['Imagenet5kMeta', 'Imagenet5k', 'Imagenet5kFiles']


class Imagenet5kMeta(object):
    """
    Provide methods to access metadata for Imagenet5k dataset.
    """

    def __init__(self, dir=None):
        if dir is None:
            raise
        self.dir = dir
        #mkdir_p(self.dir)
        self.caffepb = get_caffe_pb()
        f = os.path.join(self.dir, 'synsets.txt')
        assert(os.path.isfile(f))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))


    def get_image_list(self, name, dir_structure='original'):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`Imagenet5k.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        assert dir_structure in ['original', 'train']
        add_label_to_fname = (name != 'train' and dir_structure != 'original')
        if add_label_to_fname:
            synset = self.get_synset_1000()

        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                cls = int(cls)

                if add_label_to_fname:
                    name = os.path.join(synset[cls], name)

                ret.append((name.strip(), cls))
        assert len(ret), fname
        return ret


def _guess_dir_structure(dir):
    subdir = os.listdir(dir)[0]
    # find a subdir starting with 'n'
    if subdir.startswith('n') and \
            os.path.isdir(os.path.join(dir, subdir)):
        dir_structure = 'train'
    else:
        dir_structure = 'original'
    logger.info(
        "Assuming directory {} has {} structure.".format(
            dir, dir_structure))
    return dir_structure


class Imagenet5kFiles(RNGDataFlow):
    """
    Same as :class:`Imagenet5k`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`Imagenet5k`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'
        if dir_structure is None:
            dir_structure = _guess_dir_structure(self.full_dir)

        meta = Imagenet5kMeta(meta_dir)
        self.imglist = meta.get_image_list(name, dir_structure)

        for fname, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]


class Imagenet5k(Imagenet5kFiles):
    """
    Produces uint8 Imagenet5k images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): One of 'original' or 'train'.
                The directory structure for the 'val' directory.
                'original' means the original decompressed directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level directory structure simlar to 'dir/train/'.
                By default, it tries to automatically detect the structure.
                You probably do not need to care about this option because 'original' is what people usually have.

        Examples:

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                Imagenet5k2012_val_00000001.JPEG
                ...
              test/
                Imagenet5k2012_test_00000001.JPEG
                ...

        With the downloaded Imagenet5k_img_*.tar, you can use the following
        command to build the above structure:

        .. code-block:: none

            mkdir val && tar xvf Imagenet5k_img_val.tar -C val
            mkdir test && tar xvf Imagenet5k_img_test.tar -C test
            mkdir train && tar xvf Imagenet5k_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'

        When `dir_structure=='train'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                n01440764/
                  Imagenet5k2012_val_00000293.JPEG
                  ...
                ...
              test/
                Imagenet5k2012_test_00000001.JPEG
                ...
        """
        super(Imagenet5k, self).__init__(
            dir, name, meta_dir, shuffle, dir_structure)

    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    def get_data(self):
        for fname, label in super(Imagenet5k, self).get_data():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            if im is not None:
                yield [im, label]
            else:
                print(fname, label)

    @staticmethod
    def get_training_bbox(bbox_dir, imglist):
        import xml.etree.ElementTree as ET
        ret = []

        def parse_bbox(fname):
            root = ET.parse(fname).getroot()
            size = root.find('size').getchildren()
            size = map(int, [size[0].text, size[1].text])

            box = root.find('object').find('bndbox').getchildren()
            box = map(lambda x: float(x.text), box)
            return np.asarray(box, dtype='float32')

        with timed_operation('Loading Bounding Boxes ...'):
            cnt = 0
            for k in tqdm.trange(len(imglist)):
                fname = imglist[k][0]
                fname = fname[:-4] + 'xml'
                fname = os.path.join(bbox_dir, fname)
                try:
                    ret.append(parse_bbox(fname))
                    cnt += 1
                except Exception:
                    ret.append(None)
            logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
        return ret


try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    Imagenet5k = create_dummy_class('Imagenet5k', 'cv2')  # noqa

if __name__ == '__main__':
    meta = Imagenet5kMeta('/raid/qwang/imagenet10k/imagenet5k/meta/')

    ds = Imagenet5k('/raid/qwang/imagenet10k/imagenet5k', 'train', meta_dir='/raid/qwang/imagenet10k/imagenet5k/meta/', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break

