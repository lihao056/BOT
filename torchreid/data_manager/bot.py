from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import json
import os.path as osp
import os



class Bot(object):
    """

    """
    dataset_dir = 'bot'


    def __init__(self, root='data', verbose=True, **kwargs):
        super(Bot, self).__init__()
        '''
        osp.join就是讲a,b拼合，凑成a/b的文件夹的地址
        '''
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=False)

        num_train_imgs = len(train)
        num_test_imgs = len(test)
        num_total_imgs = num_test_imgs + num_train_imgs

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # images")
            print("  ------------------------------")
            print("  train    | {:8d}".format(num_train_imgs))
            print("  query    | {:8d}".format(num_test_imgs))
            print("  ------------------------------")
            print("  total    | {:8d}".format(num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.test = test




    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    def _process_dir(self, dir_path, relabel=False):
        dateset = []
        images_dir = os.path.join(dir_path, 'image')
        labels_dir = os.path.join(dir_path, 'label')

        for image_dir in os.listdir(images_dir):
            label_dir = os.path.join(labels_dir, image_dir.split(".")[0] + ".json")
            image_dir = os.path.join(images_dir, image_dir)

            with open(label_dir, 'r') as load_f:
                load_dict = json.load(load_f)
                for person in load_dict['annotation'][0]['object']:
                    position0 = (person['minx'], person['miny'])
                    position1 = ((person['maxx'], person['maxy']))

                    dateset.append((image_dir,
                                    position0, position1,
                                    person['gender'],
                                    person['staff'],
                                    person['customer'],
                                    person['stand'],
                                    person['sit'],
                                    person['play_with_phone']
                                    ))
        return dateset