import six.moves.cPickle as pickle
import numpy as np
import argparse
import string
import re
import random
import math
from collections import Counter
from itertools import *
parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--D_n', type=int, default=412,
                    help='number of author node')
parser.add_argument('--L_n', type=int, default=240,
                    help='number of paper node')
parser.add_argument('--M_n', type=int, default=495,
                    help='number of venue node')
parser.add_argument('--data_path', type=str, default='../data/lncRNA_disease_test/',
                    help='path to data')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')
args = parser.parse_args()
print(args)
def raw_data():

    D_max=args.D_n
    L_max = args.L_n
    M_max=args.M_n
    d_l_list = [[] for k in range(D_max)]
    l_d_list = [[] for k in range(L_max)]
    d_l_not_list=[[] for k in range(D_max)]
    raw_l_d_f = open(args.data_path + "/Raw_Dataset/lncRNA-disease.txt", "r")

    index1 = 0
    for line in raw_l_d_f:
        line = line.strip()
        disease_list = re.split(' ', line)
        for i in range(len(disease_list)):
            if (disease_list[i] == '1'):
                l_d_list[index1].append(i)
                d_l_list[i].append(index1)
            else:
                d_l_not_list[i].append(index1)
        index1 = index1 + 1
    raw_l_d_f.close()


    d_l_list_f = open(args.data_path + "total/d_l_list.txt", "w")
    l_d_list_f = open(args.data_path + "total/l_d_list.txt", "w")
    d_l_not_list_f= open(args.data_path + "total/d_l_not_list.txt", "w")
    for t in range(D_max):
        if len(d_l_list[t]):
            d_l_list_f.write(str(t) + ":")
            for tt in range(len(d_l_list[t]) - 1):
                d_l_list_f.write(str(d_l_list[t][tt]) + ",")
            d_l_list_f.write(str(d_l_list[t][-1]))
            d_l_list_f.write("\n")
    d_l_list_f.close()
    for t in range(D_max):
        if len(d_l_not_list[t]):
            d_l_not_list_f.write(str(t) + ":")
            for tt in range(len(d_l_not_list[t]) - 1):
                d_l_not_list_f.write(str(d_l_not_list[t][tt]) + ",")
            d_l_not_list_f.write(str(d_l_not_list[t][-1]))
            d_l_not_list_f.write("\n")
    d_l_not_list_f.close()
    for t in range(L_max):
        if len(l_d_list[t]):
            l_d_list_f.write(str(t) + ":")
            for tt in range(len(l_d_list[t]) - 1):
                l_d_list_f.write(str(l_d_list[t][tt]) + ",")
            l_d_list_f.write(str(l_d_list[t][-1]))
            l_d_list_f.write("\n")
    l_d_list_f.close()


    l_l_cite_list = [[] for k in range(L_max)]
    raw_l_l_f = open(args.data_path + "/Raw_Dataset/lncRNA-lncRNA.txt", "r")
    index1 = 0
    for line1 in raw_l_l_f:
        line1 = line1.strip()
        lncRNA_list1 = re.split(' ', line1)
        for i in range(len(lncRNA_list1)):
            if ((lncRNA_list1[i])>= '0.90'):
                if (index1 != i):
                    l_l_cite_list[index1].append(i)
        index1 = index1 + 1
    raw_l_l_f.close()
    l_l_cite_list_f = open(args.data_path + "total/l_l_cite_list.txt", "w")
    for t in range(L_max):
        if len(l_l_cite_list[t]):
            l_l_cite_list_f.write(str(t) + ":")
            for tt in range(len(l_l_cite_list[t]) - 1):
                l_l_cite_list_f.write(str(l_l_cite_list[t][tt]) + ",")
            l_l_cite_list_f.write(str(l_l_cite_list[t][-1]))
            l_l_cite_list_f.write("\n")
    l_l_cite_list_f.close()



    l_m_list = [[] for k in range(L_max)]
    m_l_list = [[] for k in range(M_max)]

    raw_l_m_f = open(args.data_path + "/Raw_Dataset/lncRNA-miRNA.txt", "r")

    index1 = 0
    for line in raw_l_m_f:
        line = line.strip()
        miRNA_list = re.split(' ', line)
        for i in range(len(miRNA_list)):
            if (miRNA_list[i] == '1'):
                l_m_list[index1].append(i)
                m_l_list[i].append(index1)
        index1 = index1 + 1
    raw_l_m_f.close()

    l_m_list_f = open(args.data_path + "total/l_m_list.txt", "w")
    m_l_list_f = open(args.data_path + "total/m_l_list.txt", "w")
    for t in range(L_max):
        if len(l_m_list[t]):
            l_m_list_f.write(str(t) + ":")
            for tt in range(len(l_m_list[t]) - 1):
                l_m_list_f.write(str(l_m_list[t][tt]) + ",")
            l_m_list_f.write(str(l_m_list[t][-1]))
            l_m_list_f.write("\n")
    l_m_list_f.close()

    for t in range(M_max):
        if len(m_l_list[t]):
            m_l_list_f.write(str(t) + ":")
            for tt in range(len(m_l_list[t]) - 1):
                m_l_list_f.write(str(m_l_list[t][tt]) + ",")
            m_l_list_f.write(str(m_l_list[t][-1]))
            m_l_list_f.write("\n")
    m_l_list_f.close()


raw_data()

