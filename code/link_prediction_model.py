import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve,average_precision_score


parser = argparse.ArgumentParser(description='link prediction task')
parser.add_argument('--D_n', type=int, default=412,
                        help='number of disease node')
parser.add_argument('--L_n', type=int, default=240,
                        help='number of lncRNA node')
parser.add_argument('--M_n', type=int, default=495,
                        help='number of miRNA node')
parser.add_argument('--data_path', type=str, default='../data/lncRNA_disease_test/',
                    help='path to data')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')
args = parser.parse_args()
print(args)


def load_data(data_file_name, n_features, n_samples):
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        for i, d in enumerate(data_file):
            data[i] = numpy.asarray(d[:], dtype=numpy.float)
        f.close

        return data
def model(train_num, test_num,embed_d):
    train_data_f = args.data_path + "train_feature.txt"
    train_data = load_data(train_data_f, embed_d + 3, train_num)
    train_features = train_data.astype(numpy.float32)[:, 3:-1]
    train_target = train_data.astype(numpy.int32)[:, 2]
    learner = linear_model.LogisticRegression()
    learner.fit(train_features, train_target)

    print("training finish!")

    test_data_f = args.data_path + "test_feature.txt"
    test_data = load_data(test_data_f, embed_d + 3, test_num)
    test_id = test_data.astype(numpy.int32)[:, 0:2]
    test_features = test_data.astype(numpy.float32)[:, 3:-1]
    test_target = test_data.astype(numpy.int32)[:, 2]
    test_target=numpy.array(test_target)
    predict = learner._predict_proba_lr(test_features)

    print("test prediction finish!")

    output_f = open(args.data_path + "prediction_score.txt", "w")
    for i in range(len(predict)):
        output_f.write('%d, %d, %lf\n' % (test_id[i][0], test_id[i][1], predict[i][1]))
    output_f.close();
    test_predict=[]
    for i in range(len(predict)):
        test_predict.append(predict[i][1])
    test_predict=numpy.array(test_predict)
    AUC_score = Metric.roc_auc_score(test_target, test_predict)
    lr_precision = dict()
    lr_recall = dict()
    lr_precision[0], lr_recall[0], _ = precision_recall_curve(test_target, test_predict)

    aupr = auc(sorted(lr_recall[0], reverse=True), sorted(lr_precision[0], reverse=True))

    print("AUPR: " + str(aupr))
    print("AUC: " + str(AUC_score))

    return AUC_score,aupr