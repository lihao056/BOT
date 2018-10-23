from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import json
import os.path as osp



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
        train, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        test, num_test_imgs = self._process_dir(self.test_dir, relabel=False)

        num_total_imgs = num_train_imgs + num_test_imgs

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
        img_paths = []

        for item in glob.glob(osp.join(dir_path,'image')):
            img_paths.append(glob.glob(osp.join(item,'*.jpg')))

        dataset = []
        i = 0
        for item in glob.glob(osp.join(dir_path, 'label','*')):
            img = img_paths[0][i]
            with open(item, 'r') as load_f:
                load_dict = json.load(load_f)
                object = load_dict["annotation"][0]["object"]
                for o in object:
                    minx = o['minx']
                    miny = o['miny']
                    maxx = o['maxx']
                    maxy = o['maxy']
                    position1 = [minx,miny]
                    position2 = [maxx,maxy]
                    gender = o['gender']
                    staff = o['staff']
                    customer = o['customer']
                    stand = o['stand']
                    sit = o['sit']
                    play_with_phone = o['play_with_phone']
                    dataset.append((img, position1, position2, gender, staff, customer, stand, sit, play_with_phone))
        num_imgs = len(img_paths[0])
        return dataset, num_imgs