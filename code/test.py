import re
import random
from itertools import *
import argparse
from sklearn.model_selection import KFold
import numpy as np
import link_prediction_model as LP
import matplotlib.pyplot as plt
import sklearn.metrics as Metric
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve,average_precision_score


parser = argparse.ArgumentParser(description='input data process')
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


def d_l_cite_feature_setting(train_set,test_set):
    d_embed = np.around(np.random.normal(0, 0.01, [args.D_n, args.embed_d]), 4)
    l_embed = np.around(np.random.normal(0, 0.01, [args.L_n, args.embed_d]), 4)
    embed_f = open(args.data_path + "node_embedding.txt", "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(' ', line)[0]
        if len(node_id) and (node_id[0] in ('d', 'l', 'm')):
            type_label = node_id[0]
            index = int(node_id[1:])
            embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
            if type_label == 'd':
                d_embed[index] = embed
            elif type_label == 'l':
                l_embed[index] = embed
    embed_f.close()

    train_num = 0
    d_l_list_train_feature_f = open(args.data_path + "train_feature.txt", "w")
    for h in range(len(train_set)):
        d_1 = int(train_set[h][0])
        l_2 = int(train_set[h][1])
        label = int(train_set[h][2])
        train_num += 1
        d_l_list_train_feature_f.write("%d, %d, %d," % (d_1, l_2, label))
        for d in range(args.embed_d - 1):
            d_l_list_train_feature_f.write("%f," % (2*d_embed[d_1][d] * l_embed[l_2][d]))
        d_l_list_train_feature_f.write(
             "%f" % (2*d_embed[d_1][args.embed_d - 1] * l_embed[l_2][args.embed_d - 1]))
        d_l_list_train_feature_f.write("\n")
    d_l_list_train_feature_f.close()

    test_num = 0
    d_l_list_test_feature_f = open(args.data_path + "test_feature.txt", "w")
    for h in range(len(test_set)):
        d_1 = int(test_set[h][0])
        l_2 = int(test_set[h][1])
        label = int(test_set[h][2])
        test_num += 1
        d_l_list_test_feature_f.write("%d, %d, %d," % (d_1, l_2, label))
        for d in range(args.embed_d - 1):
            d_l_list_test_feature_f.write("%f," % (2*d_embed[d_1][d] * l_embed[l_2][d]))
        d_l_list_test_feature_f.write("%f" % (2*d_embed[d_1][args.embed_d - 1] * l_embed[l_2][args.embed_d - 1]))
        d_l_list_test_feature_f.write("\n")
    d_l_list_test_feature_f.close()

    print("d_l_cite_train_num: " + str(train_num))
    print("d_l_cite_test_num: " + str(test_num))
    return train_num, test_num

def d_l_predict():

    d_l_not_list = [[] for k in range(args.D_n)]
    f_name = "d_l_not_list.txt"
    neigh_f = open(args.data_path + "total/" + f_name, "r")
    for line in neigh_f:
        line = line.strip()
        node_id = int(re.split(':', line)[0])
        neigh_list = re.split(':', line)[1]
        neigh_list_id = re.split(',', neigh_list)
        for j in range(len(neigh_list_id)):
            d_l_not_list[node_id].append(neigh_list_id[j])
    neigh_f.close()
    d_l_list = [[] for k in range(args.D_n)]
    f_name = "d_l_list.txt"
    neigh_f = open(args.data_path + "total/" + f_name, "r")
    for line in neigh_f:
        line = line.strip()
        node_id = int(re.split(':', line)[0])
        neigh_list = re.split(':', line)[1]
        neigh_list_id = re.split(',', neigh_list)
        for j in range(len(neigh_list_id)):
            d_l_list[node_id].append(neigh_list_id[j])
    neigh_f.close()

    d_l_list_total_f = open(args.data_path + "d_l_list_total.txt", "w")
    for i in range(len(d_l_list)):
        for j in range(len(d_l_list[i])):
            l_id = d_l_list[i][j]
            d_l_list_total_f.write("%d, %d, %d\n" % (i, int(l_id), 1))
            nl_id = random.choice(d_l_not_list[i])
            d_l_list_total_f.write("%d, %d, %d\n" % (i, int(nl_id), 0))
    d_l_list_total_f.close()
    fname = open(args.data_path + "/d_l_list_total.txt", "r")
    total=[]
    for line in fname:
        line.strip()
        d_id = int(re.split(',', line)[0])
        l_id=int(re.split(',', line)[1])
        label=int(re.split(',', line)[2])
        total.append([d_id,l_id,label])
    fname.close()
    total=np.array(total)
    fold=5
    kf=KFold(n_splits=fold,shuffle=False)
    i=0
    auc_total=0
    aupr_total=0

    for train_index,test_index in kf.split(total):
        i=i+1
        print(i)

        train_set=total[train_index]
        test_set=total[test_index]

        train_num, test_num = d_l_cite_feature_setting(train_set,test_set)
        auc1,aupr=LP.model(train_num, test_num,args.embed_d)
        auc_total+=auc1
        aupr_total+=aupr


    print("------lncRNA disease association  prediction  end------")
    print("AUC: " + str(auc_total/fold))

d_l_predict()
