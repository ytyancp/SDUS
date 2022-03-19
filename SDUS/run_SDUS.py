# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/17 15:34
@Auth ： zhensu1998
@File ：run_SDUS.py
@IDE ：PyCharm
@Mail：2286583124@qq.com

"""
import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import settings
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
import methods_SDUS1
import methods_SDUS2


dataPath = settings.dataPath
listFile = os.listdir(dataPath)


def evaluation(estimators, test_X, test_Y):
    """
    [g_mean, auc, f1]
    """
    score = []
    for key, values in estimators.items():
        pre_Y = []
        for estimator in values:
            pre_Y.append(estimator.predict(test_X))
        pre_Y = np.where(np.mean(pre_Y, axis=0) < 0.5, 0, 1)
        auc = roc_auc_score(test_Y, pre_Y)
        f1 = f1_score(test_Y, pre_Y, pos_label=0)
        g_mean = geometric_mean_score(test_Y, pre_Y, pos_label=0)
        score.append(g_mean)
        score.append(auc)
        score.append(f1)
    return np.array(score)

def fitCorss(file):
    """
    :param file: file name
    :return:
    {'file': file, 'score': [scoreMean_SDUS1, scoreMean_SDUS2], 'std': [stdScore_SDUS1, stdScore_SDUS2]}
    """
    print(file)
    data = np.loadtxt(dataPath + '\\' + file, delimiter=',')
    np.random.shuffle(data)
    X, Y = data[:, :-1], data[:, -1]
    finalScore_SDUS1 = []
    finalScore_SDUS2 = []
    for loop in tqdm(range(settings.crossCount), desc=file):
        skf = StratifiedKFold(n_splits=settings.n_splits, shuffle=True)
        scores_SDUS1 = []
        scores_SDUS2 = []
        for train_index, test_index in skf.split(X, Y):
            train_X, train_Y = X[train_index], Y[train_index]
            test_X, test_Y = X[test_index], Y[test_index]
            estimators_SDUS1 = methods_SDUS1.methods().run(train_X, train_Y)
            score_SDUS1 = evaluation(estimators_SDUS1, test_X, test_Y)
            estimators_SDUS2 = methods_SDUS2.methods().run(train_X, train_Y)
            score_SDUS2 = evaluation(estimators_SDUS2, test_X, test_Y)
            scores_SDUS1.append(score_SDUS1)
            scores_SDUS2.append(score_SDUS2)
        finalScore_SDUS1.append(np.mean(scores_SDUS1, axis=0))
        finalScore_SDUS2.append(np.mean(scores_SDUS2, axis=0))
    stdScore_SDUS1 = np.std(finalScore_SDUS1, axis=0, ddof=1)
    scoreMean_SDUS1 = np.mean(finalScore_SDUS1, axis=0)
    stdScore_SDUS2 = np.std(finalScore_SDUS2, axis=0, ddof=1)
    scoreMean_SDUS2 = np.mean(finalScore_SDUS2, axis=0)
    result = {'file': file, 'score': [scoreMean_SDUS1, scoreMean_SDUS2], 'std': [stdScore_SDUS1, stdScore_SDUS2]}
    return result


if __name__ == '__main__':
    pool = multiprocessing.Pool(10)
    results = []
    for file in listFile:
        result = pool.apply_async(fitCorss, args=[file])
        results.append(result)
    pool.close()
    pool.join()
    for res_dic in results:
        res = res_dic.get()
        file = res['file']
        scoreMean_SDUS1 = res['score'][0]
        scoreMean_SDUS2 = res['score'][1]
        stdScore_SDUS1 = res['std'][0]
        stdScore_SDUS2 = res['std'][1]

        with open(settings.savePathSDUS1score, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(scoreMean_SDUS1)
        with open(settings.savePathSDUS1std, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stdScore_SDUS1)
        with open(settings.savePathSDUS2score, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(scoreMean_SDUS2)
        with open(settings.savePathSDUS2std, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stdScore_SDUS2)
