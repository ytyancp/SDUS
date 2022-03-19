# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/17 14:44
@Auth ： zhensu1998
@File ：SDUS_SPN.py
@IDE ：PyCharm
@Mail：2286583124@qq.com

"""

import numpy as np
import copy
# 0：minority class
# 1：majority class
class SDUS_SPN:
    def __init__(self):
        self.X = None  # training set
        self.y = None  # training label
        self.I = dict()
        self.covered = np.array([])  # whether constructed, 1:constructed, 0:Unconstructed
        self.SPN_infor = list()

    def initial(self):
        for index, value in enumerate(self.y):
            if value not in self.I.keys():
                self.I[value] = list()
            self.I[value].append(index)
        self.I = dict(sorted(self.I.items(), key=lambda x: x[0], reverse=True))

    def distances(self, f1, f2):
        dist = np.sqrt(np.sum((f1 - f2) ** 2))
        return dist

    def d1(self, lab, s):    # find the closet minority
        m = (lab + 1) % len(self.I)
        d1 = self.distances(self.X[s], self.X[self.I[m][0]])
        for key in self.I.keys():
            if key != lab:
                for i in self.I[key]:
                    temp = self.distances(self.X[i], self.X[s])
                    if temp < d1:
                        d1 = temp
        return d1

    def d2(self, lab, s, d1):  # find the farthest majority
        d2 = self.distances(self.X[s], self.X[s])
        a_index = s
        for index in self.I[lab]:
            temp = self.distances(self.X[s], self.X[index])
            if temp < d1:
                if d2 < temp:
                    d2 = temp
                    a_index = index
        return d2, a_index

    def cover_sample(self, s, lab, d, A_index):     # construct a SPN
        SPN = {'center': [s], 'id': [], 'a_index':[]}
        SPN['a_index'].append(A_index)
        for i in self.I[lab]:
            if self.covered[i] == 1:
                continue
            else:
                if (i == s):
                    self.covered[i] = 1
                    SPN['id'].append(i)
                    continue
                if self.distances(self.X[s], self.X[i]) <= d:
                    SPN['id'].append(i)
                    self.covered[i] = 1
        return SPN

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.covered = np.zeros(len(self.X))
        self.initial()
        lab = 1
        V = copy.deepcopy(self.I[lab])
        while (len(V) > 0):
            s = np.random.choice(V)
            if self.covered[s] == 1:
                V.remove(s)
                continue
            d1 = self.d1(lab, s)
            d2, a_index = self.d2(lab, s, d1)
            SPN = self.cover_sample(s, lab, d2, a_index)
            self.SPN_infor.append(SPN)
        return self.SPN_infor