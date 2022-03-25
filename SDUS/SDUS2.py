# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/17 15:10
@Auth ： zhensu1998
@File ：SDUS2.py
@IDE ：PyCharm
@Mail：2286583124@qq.com

"""

from collections import Counter
import logging
from copy import deepcopy
from SDUS_SPN import SDUS_SPN
import numpy as np
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')


# 0:the minority class
# 1:the majority class
class SDUS2:
    def __init__(self):
        self.X = None
        self.y = None
        self.Sinfor = None  # SPN information
        self.S_maj = []    # Indicates an SPN that needs to be independently selected
        self.S_rest = []   # Indicates an SPN that needs to be aggregated for selection
        self.maj_num = None
        self.min_num = None
        self.ID = []

    def cos_sim(self, vec1, vec2):
        cos_sim_dist = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return cos_sim_dist

    def sin_sim(self, hudu):
        return np.sin(hudu)

    # Dividing the SPN into four parts
    def field_list(self, S):
        up_right_set = []
        up_left_set = []
        down_right_set = []
        down_left_set = []
        o_id = S['center'][0]
        a_id = S['a_index'][0]
        # 向量oa
        o_a_vec = self.X[a_id] - self.X[o_id]
        up_set = []
        down_set = []
        min_cos = 1.5
        max_cos = -1.5
        min_cos_id = -1
        max_cos_id = -1
        for B_id in S['id']:
            o_b_vec = self.X[B_id] - self.X[o_id]
            cos_sim = self.cos_sim(o_a_vec, o_b_vec)
            if cos_sim >= 0:
                if cos_sim < min_cos:
                    min_cos = cos_sim
                    min_cos_id = B_id
                up_set.append(B_id)
            else:
                if cos_sim > max_cos:
                    max_cos = cos_sim
                    max_cos_id = B_id
                down_set.append(B_id)
        hudu_90 = 1.5707963267948966
        hudu_min_cos = np.arccos(min_cos)
        hudu_min_cha = hudu_90 - hudu_min_cos
        o_c_vec = self.X[min_cos_id] - self.X[o_id]
        hudu_max_cos = np.arccos(max_cos)
        hudu_max_cha = hudu_max_cos - hudu_90
        o_d_vec = self.X[max_cos_id] - self.X[o_id]
        for S_id in up_set:
            S_vec = self.X[S_id] - self.X[o_id]
            cos_sim = self.cos_sim(S_vec, o_c_vec)
            zhenshihudu = np.arccos(cos_sim) + hudu_min_cha
            if zhenshihudu <= hudu_90:
                up_right_set.append(S_id)
            else:
                up_left_set.append(S_id)
        for S_id in down_set:
            S_vec = self.X[S_id] - self.X[o_id]
            cos_sim = self.cos_sim(S_vec, o_d_vec)
            zhenshihudu = np.arccos(cos_sim) + hudu_max_cha
            if zhenshihudu <= hudu_90:
                down_left_set.append(S_id)
            else:
                down_right_set.append(S_id)
        return dict({'up_right_set': up_right_set, 'down_right_set': down_right_set, 'up_left_set': up_left_set,
                     'down_left_set': down_left_set})

    def get_if(self, a, b):
        if a < b:
            return True
        else:
            return False

    def distance(self, ma, mb):
        l = len(ma)
        abt = np.matrix(ma) * np.matrix(mb.T)
        sqa = np.tile(np.sum(ma ** 2, axis=1), l).reshape((l, l)).T
        sqb = np.tile(np.sum(mb ** 2, axis=1), l).reshape((l, l))
        dis_matrix = np.sqrt(sqa + sqb - 2 * abt + 0.0001)
        return np.array(dis_matrix)


    def weighted_under_sampling(self, id, weight, num):
        select_index = np.random.choice(id, size=num, replace=False, p=weight)
        return select_index

    def weight_norm(self, matrix):
        sum_ = np.sum(matrix)
        matrix = list([x / sum_ for x in matrix])
        return matrix

    def fit(self, X, y):
        self.X = X
        self.y = y
        dic = Counter(self.y)
        spn = SDUS_SPN()
        self.Sinfor = spn.fit(self.X, self.y)
        self.maj_num, self.min_num = dic[1], dic[0]

        for S in self.Sinfor:
            if (len(S['id']) / self.maj_num) * self.min_num >= 1:
                self.S_maj.append(S)
            else:
                self.S_rest.append(S)

        # Whether there is an SPN with the number of samples to be sampled greater than or equal to 1 within the SPN
        if len(self.S_maj) != 0:
            select_num = 0
            for S in self.S_maj:
                id_num = len(S['id'])
                sel_num = round(id_num / self.maj_num * self.min_num)
                select_num = select_num + sel_num
                S['id'].remove(S['center'][0])
                sel_id = deepcopy(S['center'])
                dic_set = self.field_list(S)
                sort_len_set = sorted(list(
                    zip([len(dic_set['up_right_set']), len(dic_set['down_right_set']), len(dic_set['up_left_set']),
                         len(dic_set['down_left_set'])],
                        ['up_right_set', 'down_right_set', 'up_left_set', 'down_left_set'])), key=lambda x: x[0],
                    reverse=True)

                num = 1
                # Select samples in a round fashion across four regions
                while True:
                    for i in range(len(sort_len_set)):
                        if sort_len_set[i][0] > 0:
                            temp_id = np.random.choice(dic_set[sort_len_set[i][1]])
                            dic_set[sort_len_set[i][1]].remove(temp_id)
                            sort_len_set[i] = (sort_len_set[i][0] - 1, sort_len_set[i][1])
                            sel_id.append(temp_id)
                            num += 1
                        if self.get_if(sel_num - 1, num):
                            break
                    if self.get_if(sel_num - 1, num):
                        break
                self.ID.extend(sel_id)

            # # If the number of selected classes is less than the number of minority classes then continue to select from the aggregated SPNs
            if select_num < self.min_num:
                if len(self.S_rest) != 0:
                    sample_set_rest_index = []
                    for S in self.S_rest:
                        for index in S['id']:
                            sample_set_rest_index.append(index)
                    sample_set_rest = self.X[sample_set_rest_index]
                    select_num_rest = round(len(sample_set_rest) / self.maj_num * self.min_num)
                    dis_matrix_rest = self.distance(sample_set_rest, sample_set_rest)
                    sum_matrix_rest = np.sum(dis_matrix_rest, axis=1)
                    sum_matrix_rest = 1 / sum_matrix_rest
                    sum_matrix_rest = self.weight_norm(sum_matrix_rest)
                    weights = sum_matrix_rest
                    sampled = self.weighted_under_sampling(sample_set_rest_index, weights, select_num_rest)
                    for temp_sampled_index in sampled:
                        self.ID.append(temp_sampled_index)

            re_maj = np.c_[self.X[np.array(self.ID)], np.ones(len(self.ID))]
            re_min = np.c_[self.X[np.where(self.y == 0)[0]], np.zeros(self.min_num)]
            resample = np.r_[re_maj, re_min]
            return resample

        else:  # if none of the SPNs need to sample more than 1 sample
            sample_set_index = []
            for S in self.S_rest:
                for index in S['id']:
                    sample_set_index.append(index)
            sample_set = self.X[sample_set_index]
            select_num = self.min_num
            dis_matrix = self.distance(sample_set, sample_set)
            sum_matrix = np.sum(dis_matrix, axis=1)
            sum_matrix = 1 / sum_matrix
            sum_matrix = self.weight_norm(sum_matrix)
            weights = sum_matrix
            sampled = self.weighted_under_sampling(sample_set_index, weights, select_num)
            for temp_sampled_index in sampled:
                self.ID.append(temp_sampled_index)

            re_maj = np.c_[self.X[np.array(self.ID)], np.ones(len(self.ID))]
            re_min = np.c_[self.X[np.where(self.y == 0)[0]], np.zeros(self.min_num)]
            resample = np.r_[re_maj, re_min]
            return resample
