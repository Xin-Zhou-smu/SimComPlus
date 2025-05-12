import math
from itertools import groupby
import csv
from collections import Counter
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score, \
    accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from matplotlib import pyplot
import argparse
import time

np.random.seed(10)


def read_csv_1(fname):
    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(int(line[0]))
            la.append(int(line[1]))
            pred.append(float(line[2]))

    # print(len(pred), len(label), len(la))
    return pred, label, la


def read_csv_2(fname):
    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))

    # print(len(pred), len(label))
    return pred, label


def eval_(y_true, y_pred, thresh=None):
    # print('size:', len(y_true), len(y_pred))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    # auc_pc(y_true, y_pred)
    if thresh != None:
        y_pred = [1.0 if p > thresh else 0.0 for p in y_pred]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('AUC:', auc)


def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    return lr_auc


parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='jdt')
parser.add_argument('-combiner', type=str, default='average')
args = parser.parse_args()

data_dir1 = "./Com/pred_scores/"
data_dir2 = "./Sim/pred_scores/"
data_dir3 = "./Early_fusion/"

project = args.project


# Com
com_ = data_dir1 + 'test_com_' + args.project + '.csv'


# Sim
sim_ = data_dir2 + 'test_sim_' + args.project + '.csv'


# Gating mechanism fusion
if args.project == 'openstack':
    lr = 0.0001
elif args.project == 'qt':
    lr = 8e-05
elif args.project == 'jdt':
    lr = 2e-05
elif args.project == 'gerrit':
    lr = 5e-05
elif args.project == 'go':
    lr = 8e-05
elif args.project == 'platform':
    lr = 0.0002
tabular_ = data_dir3 + 'model/'+ args.project +'/gating_on_cat_and_num_feats_then_sum_quantile_normal_ohe_lr'+str(lr)+'_seed42/test_com_score.csv'


## Sim
pred1, label1 = read_csv_2(sim_)


## Com
pred2, label2 = read_csv_2(com_)


## Early Fusion Model
pred3, label3 = read_csv_2(tabular_)







if args.combiner == 'average':
    pred_fusion3 = [(pred1[i] + pred2[i] + pred3[i]) / 3 for i in range(len(pred1))]

def report_metric(pred, label):
    label = [float(l) for l in label]
    auc_ = roc_auc_score(y_true=np.array(label), y_score=np.array(pred))
    pc_ = auc_pc(label, pred)
    real_label = [float(l) for l in label]
    real_pred = [1 if p > 0.5 else 0 for p in pred]
    f1_ = f1_score(y_true=real_label, y_pred=real_pred)
    print("AUC-ROC:{}  AUC-PR:{}  F1-Score:{}".format(auc_, pc_, f1_))
    return auc_, pc_, f1_



roc, pr, f1s = report_metric(pred_fusion3, label1)
