from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave


class Chart(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'chart'
    LabelQuery = {
        'AreaGraph':0,
        'BarGraph': 1,
        'LineGraph': 2,
        'Map': 3,
        'ParetoChart': 4,
        'PieChart': 5,
        'RadarPlot': 6,
        'ScatterGraph': 7,
        'Table': 8,
        'VennDiagram': 9
    }

    def __init__(self, root='data', verbose=True, **kwargs):
        super(Chart, self).__init__()
        '''
        osp.join就是讲a,b拼合，凑成a/b的文件夹的地址
        '''
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        cls = len(self.LabelQuery)
        train, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        test, num_test_imgs = self._process_dir(self.test_dir, relabel=False)

        num_total_imgs = num_train_imgs + num_test_imgs

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(cls, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(cls, num_test_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(cls, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.test = test

        self.cls = cls


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = []
        for item in glob.glob(osp.join(dir_path,'*')):
            img_paths.append(glob.glob(osp.join(item,'*.jpg')))

        dataset = []
        for i in range(len(img_paths)):
            for img_path in img_paths[i]:
                dataset.append((img_path, i))

        num_imgs = len(dataset)
        return dataset, num_imgs