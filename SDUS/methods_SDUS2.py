# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/17 15:43
@Auth ： zhensu1998
@File ：methods_SDUS2.py
@IDE ：PyCharm
@Mail：2286583124@qq.com

"""

import numpy as np
import logging
import copy
import settings
import SDUS2
from sklearn import ensemble

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')

class methods(object):
    def __init__(self):
        self.estimators = copy.deepcopy(settings.classifier)

    def RF(self, resample):
        sel_X, sel_y = resample[:, :-1], resample[:, -1]
        cls = ensemble.RandomForestClassifier()
        cls.fit(sel_X, sel_y)
        return cls

    def run(self, X, Y):
        self.SDUS2(X, Y)
        return self.estimators

    def SDUS2(self, X, Y):
        for loop in range(settings.trainingCount):
            sdus2 = SDUS2.SDUS2()
            resample = sdus2.fit(X, Y)
            self.estimators['RF'].append(self.RF(resample))
        return self.estimators