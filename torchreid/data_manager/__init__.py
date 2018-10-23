from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .bot import Bot


__imgreid_factory = {
    'bot': Bot
}



def get_names():
    return list(__imgreid_factory.keys())

'''
    return __imgreid_factory[name](**kwargs)
    这句话直接将模型参数等激活了
'''
def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

