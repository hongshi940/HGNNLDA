import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
import argparse

class input_data(object):
    def __init__(self, args):
        self.args = args
        d_l_list = [[] for k in range(self.args.D_n)]
        l_d_list = [[] for k in range(self.args.L_n)]
        l_l_cite_list = [[] for k in range(self.args.L_n)]
        l_m_list = [[] for k in range(self.args.L_n)]
        m_l_list = [[] for k in range(self.args.M_n)]
        ff = ["d_l_list.txt", "l_d_list.txt", "l_l_cite_list.txt", "m_l_list.txt",
              "l_m_list.txt"]
        for i in range(len(ff)):
            f_name = ff[i]
            neigh_f = open(self.args.data_path+ "/total/"+f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if f_name == 'd_l_list.txt':
                    for j in range(len(neigh_list_id)):
                        d_l_list[node_id].append('l' + str(neigh_list_id[j]))
                elif f_name == 'l_d_list.txt':
                    for j in range(len(neigh_list_id)):
                        l_d_list[node_id].append('d' + str(neigh_list_id[j]))
                elif f_name == 'l_l_cite_list.txt':
                    for j in range(len(neigh_list_id)):
                        l_l_cite_list[node_id].append('l' + str(neigh_list_id[j]))
                elif f_name == 'm_l_list.txt':
                    for j in range(len(neigh_list_id)):
                        m_l_list[node_id].append('l' + str(neigh_list_id[j]))
                else:
                    for j in range(len(neigh_list_id)):
                        l_m_list[node_id].append('m' + str(neigh_list_id[j]))
            neigh_f.close()
        l_neigh_list = [[] for k in range(self.args.L_n)]
        for i in range(self.args.L_n):
            l_neigh_list[i] += l_d_list[i]
            l_neigh_list[i] += l_l_cite_list[i]
            l_neigh_list[i] += l_m_list[i]

        self.d_l_list_total = d_l_list
        self.l_d_list_total = l_d_list
        self.l_l_cite_list_total = l_l_cite_list
        self.l_m_list_total = l_m_list
        self.m_l_list_total = m_l_list
        self.l_neigh_list_total = l_neigh_list

        self.het_walk_restart()

        if self.args.train_test_label != 2:
            self.triple_sample_p = self.compute_sample_p()

            l_embed1 = np.zeros((self.args.L_n, self.args.in_f_d))
            l_embed2 = np.zeros((self.args.L_n, self.args.in_f_d))
            self.l_embed1 = l_embed1
            self.l_embed2 = l_embed2


            d_net_embed = np.zeros((self.args.D_n, self.args.in_f_d))
            l_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
            m_net_embed = np.zeros((self.args.M_n, self.args.in_f_d))
            net_e_f = open(self.args.data_path + "node_net_embedding_total.txt", "r")
            for line in islice(net_e_f, 1, None):
                line = line.strip()
                index = re.split(' ', line)[0]
                if len(index) and (index[0] == 'd' or index[0] == 'l' or index[0] == 'm'):
                    embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
                    if index[0] == 'd':
                        d_net_embed[int(index[1:])] = embeds
                    elif index[0] == 'm':
                        m_net_embed[int(index[1:])] = embeds
                    else:
                        l_net_embed[int(index[1:])] = embeds
            net_e_f.close()

            l_m_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
            for i in range(self.args.L_n):
                if len(l_m_list[i]):
                    for j in range(len(l_m_list[i])):
                        m_id = int(l_m_list[i][j][1:])
                        l_m_net_embed[i] = np.add(l_m_net_embed[i], m_net_embed[m_id])
                    l_m_net_embed[i] = l_m_net_embed[i] / len(l_m_list[i])

            l_d_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
            for i in range(self.args.L_n):
                if len(l_d_list[i]):
                    for j in range(len(l_d_list[i])):
                        d_id = int(l_d_list[i][j][1:])
                        l_d_net_embed[i] = np.add(l_d_net_embed[i], d_net_embed[d_id])
                    l_d_net_embed[i] = l_d_net_embed[i] / len(l_d_list[i])

            l_ref_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
            for i in range(self.args.L_n):
                if len(l_l_cite_list[i]):
                    for j in range(len(l_l_cite_list[i])):
                        l_id = int(l_l_cite_list[i][j][1:])
                        l_ref_net_embed[i] = np.add(l_ref_net_embed[i], l_net_embed[l_id])
                    l_ref_net_embed[i] = l_ref_net_embed[i] / len(l_l_cite_list[i])
                else:
                    l_ref_net_embed[i] = l_net_embed[i]


            d_text_embed = np.zeros((self.args.D_n, self.args.in_f_d * 3))
            for i in range(self.args.D_n):
                if len(d_l_list[i]):
                    feature_temp = []
                    if len(d_l_list[i]) >= 3:
                        for j in range(3):
                            feature_temp.append(l_embed1[int(d_l_list[i][j][1:])])
                    else:
                        for j in range(len(d_l_list[i])):
                            feature_temp.append(l_embed1[int(d_l_list[i][j][1:])])
                        for k in range(len(d_l_list[i]), 3):
                            feature_temp.append(l_embed1[int(d_l_list[i][-1][1:])])

                    feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
                    d_text_embed[i] = feature_temp


            m_text_embed = np.zeros((self.args.M_n, self.args.in_f_d * 5))
            for i in range(self.args.M_n):
                if len(m_l_list[i]):
                    feature_temp = []
                    if len(m_l_list[i]) >= 5:
                        for j in range(5):
                            feature_temp.append(l_embed1[int(m_l_list[i][j][1:])])
                    else:
                        for j in range(len(m_l_list[i])):
                            feature_temp.append(l_embed1[int(m_l_list[i][j][1:])])
                        for k in range(len(m_l_list[i]), 5):
                            feature_temp.append(l_embed1[int(m_l_list[i][-1][1:])])

                    feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
                    m_text_embed[i] = feature_temp

            self.l_m_net_embed = l_m_net_embed
            self.l_d_net_embed = l_d_net_embed
            self.l_ref_net_embed = l_ref_net_embed
            self.l_net_embed = l_net_embed
            self.d_net_embed = d_net_embed
            self.d_text_embed = d_text_embed
            self.m_net_embed = m_net_embed
            self.m_text_embed = m_text_embed

            # store neighbor set from random walk sequence
            d_neigh_list = [[[] for i in range(self.args.D_n)] for j in range(3)]
            l_neigh_list = [[[] for i in range(self.args.L_n)] for j in range(3)]
            m_neigh_list = [[[] for i in range(self.args.M_n)] for j in range(3)]

            het_neigh_train_f = open(self.args.data_path + "het_neigh_total.txt", "r")
            for line in het_neigh_train_f:
                line = line.strip()
                node_id = re.split(':', line)[0]
                neigh = re.split(':', line)[1]
                neigh_list = re.split(',', neigh)
                if node_id[0] == 'd' and len(node_id) > 1:
                    for j in range(len(neigh_list)):
                        if neigh_list[j][0] == 'd':
                            d_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'l':
                            d_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        elif neigh_list[j][0] == 'm':
                            d_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                elif node_id[0] == 'l' and len(node_id) > 1:
                    for j in range(len(neigh_list)):
                        if neigh_list[j][0] == 'd':
                            l_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        if neigh_list[j][0] == 'l':
                            l_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        if neigh_list[j][0] == 'm':
                            l_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
                elif node_id[0] == 'm' and len(node_id) > 1:
                    for j in range(len(neigh_list)):
                        if neigh_list[j][0] == 'd':
                            m_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        if neigh_list[j][0] == 'l':
                            m_neigh_list[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
                        if neigh_list[j][0] == 'm':
                            m_neigh_list[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
            het_neigh_train_f.close()


            # store top neighbor set (based on frequency) from random walk sequence
            d_neigh_list_top = [[[] for i in range(self.args.D_n)] for j in range(3)]
            l_neigh_list_top = [[[] for i in range(self.args.L_n)] for j in range(3)]
            m_neigh_list_top = [[[] for i in range(self.args.M_n)] for j in range(3)]
            top_k = [10, 10, 3]  # fix each neighor type size
            for i in range(self.args.D_n):
                for j in range(3):
                    d_neigh_list_train_temp = Counter(d_neigh_list[j][i])
                    top_list = d_neigh_list_train_temp.most_common(top_k[j])
                    neigh_size = 0
                    if j == 0 or j == 1:
                        neigh_size = 10
                    else:
                        neigh_size = 3
                    for k in range(len(top_list)):
                        d_neigh_list_top[j][i].append(int(top_list[k][0]))
                    if len(d_neigh_list_top[j][i]) and len(d_neigh_list_top[j][i]) < neigh_size:
                        for l in range(len(d_neigh_list_top[j][i]), neigh_size):
                            d_neigh_list_top[j][i].append(random.choice(d_neigh_list_top[j][i]))

            for i in range(self.args.L_n):
                for j in range(3):
                    l_neigh_list_train_temp = Counter(l_neigh_list[j][i])
                    top_list = l_neigh_list_train_temp.most_common(top_k[j])
                    neigh_size = 0
                    if j == 0 or j == 1:
                        neigh_size = 10
                    else:
                        neigh_size = 3
                    for k in range(len(top_list)):
                        l_neigh_list_top[j][i].append(int(top_list[k][0]))
                    if len(l_neigh_list_top[j][i]) and len(l_neigh_list_top[j][i]) < neigh_size:
                        for l in range(len(l_neigh_list_top[j][i]), neigh_size):
                            l_neigh_list_top[j][i].append(random.choice(l_neigh_list_top[j][i]))

            for i in range(self.args.M_n):
                for j in range(3):
                    m_neigh_list_train_temp = Counter(m_neigh_list[j][i])
                    top_list = m_neigh_list_train_temp.most_common(top_k[j])
                    neigh_size = 0
                    if j == 0 or j == 1:
                        neigh_size = 10
                    else:
                        neigh_size = 3
                    for k in range(len(top_list)):
                        m_neigh_list_top[j][i].append(int(top_list[k][0]))
                    if len(m_neigh_list_top[j][i]) and len(m_neigh_list_top[j][i]) < neigh_size:
                        for l in range(len(m_neigh_list_top[j][i]), neigh_size):
                            m_neigh_list_top[j][i].append(random.choice(m_neigh_list_top[j][i]))

            d_neigh_list[:] = []
            l_neigh_list[:] = []
            m_neigh_list[:] = []

            self.d_neigh_list_train = d_neigh_list_top
            self.l_neigh_list_train = l_neigh_list_top
            self.m_neigh_list_train = m_neigh_list_top


            train_id_list = [[] for i in range(3)]
            for i in range(3):
                if i == 0:
                    for l in range(self.args.D_n):
                        if len(d_neigh_list_top[i][l]):
                            train_id_list[i].append(l)
                    self.d_train_id_list = np.array(train_id_list[i])
                elif i == 1:
                    for l in range(self.args.L_n):
                        if len(l_neigh_list_top[i][l]):
                            train_id_list[i].append(l)
                    self.l_train_id_list = np.array(train_id_list[i])
                else:
                    for l in range(self.args.M_n):
                        if len(m_neigh_list_top[i][l]):
                            train_id_list[i].append(l)
                    self.m_train_id_list = np.array(train_id_list[i])



    def gen_het_rand_walk(self):
        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "w")
        for i in range(self.args.walk_n):
            for j in range(self.args.D_n):
                if len(self.d_l_list_total[j]):
                    curNode = "d" + str(j)
                    het_walk_f.write(curNode + " ")
                    for l in range(self.args.walk_L - 1):
                        if curNode[0] == "d":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.d_l_list_total[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "l":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.l_neigh_list_total[curNode])
                            het_walk_f.write(curNode + " ")
                        elif curNode[0] == "m":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.m_l_list_total[curNode])
                            het_walk_f.write(curNode + " ")
                    het_walk_f.write("\n")
        het_walk_f.close()

    def het_walk_restart(self):

        d_neigh_list_total = [[] for k in range(self.args.D_n)]
        l_neigh_list_total = [[] for k in range(self.args.L_n)]
        m_neigh_list_total = [[] for k in range(self.args.M_n)]

        # generate neighbor set via random walk with restart
        node_n = [self.args.D_n, self.args.L_n, self.args.M_n]
        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_temp = self.d_l_list_total[j]
                    neigh_train = d_neigh_list_total[j]
                    curNode = "d" + str(j)
                elif i == 1:
                    neigh_temp = self.l_neigh_list_total[j]
                    neigh_train = l_neigh_list_total[j]
                    curNode = "l" + str(j)
                else:
                    neigh_temp = self.m_l_list_total[j]
                    neigh_train = m_neigh_list_total[j]
                    curNode = "m" + str(j)
                if len(neigh_temp):
                    neigh_L = 0
                    d_L = 0
                    l_L = 0
                    m_L = 0
                    while neigh_L < 100:  # maximum neighbor size = 100
                        rand_p = random.random()  # return p
                        if rand_p > 0.5:
                            if curNode[0] == "d":
                                curNode = random.choice(self.d_l_list_total[int(curNode[1:])])
                                if l_L < 46:  # size constraint (make sure each type of neighobr is sampled)
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    l_L += 1
                            elif curNode[0] == "l":
                                curNode = random.choice(self.l_neigh_list_total[int(curNode[1:])])
                                if curNode != ('d' + str(j)) and curNode[0] == 'd' and d_L < 46:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    d_L += 1
                                elif curNode[0] == 'm':
                                    if m_L < 11:
                                        neigh_train.append(curNode)
                                        neigh_L += 1
                                        m_L += 1
                            elif curNode[0] == "m":
                                curNode = random.choice(self.m_l_list_total[int(curNode[1:])])
                                if l_L < 46:
                                    neigh_train.append(curNode)
                                    neigh_L += 1
                                    l_L += 1
                        else:
                            if i == 0:
                                curNode = ('d' + str(j))
                            elif i == 1:
                                curNode = ('l' + str(j))
                            else:
                                curNode = ('m' + str(j))

        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    d_neigh_list_total[j] = list(d_neigh_list_total[j])
                elif i == 1:
                    l_neigh_list_total[j] = list(l_neigh_list_total[j])
                else:
                    m_neigh_list_total[j] = list(m_neigh_list_total[j])

        neigh_f = open(self.args.data_path + "het_neigh_total.txt", "w")
        for i in range(3):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_train = d_neigh_list_total[j]
                    curNode = "d" + str(j)
                elif i == 1:
                    neigh_train = l_neigh_list_total[j]
                    curNode = "l" + str(j)
                else:
                    neigh_train = m_neigh_list_total[j]
                    curNode = "m" + str(j)
                if len(neigh_train):
                    neigh_f.write(curNode + ":")
                    for k in range(len(neigh_train) - 1):
                        neigh_f.write(neigh_train[k] + ",")
                    neigh_f.write(neigh_train[-1] + "\n")
        neigh_f.close()

    def compute_sample_p(self):
        print("computing sampling ratio for each kind of triple ...")
        window = self.args.window
        walk_L = self.args.walk_L
        D_n = self.args.D_n
        L_n = self.args.L_n
        M_n = self.args.M_n

        total_triple_n = [0.0] * 9  # nine kinds of triples
        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
        centerNode = ''
        neighNode = ''

        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'd':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd':
                                    total_triple_n[0] += 1
                                elif neighNode[0] == 'l':
                                    total_triple_n[1] += 1
                                elif neighNode[0] == 'm':
                                    total_triple_n[2] += 1
                    elif centerNode[0] == 'l':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd':
                                    total_triple_n[3] += 1
                                elif neighNode[0] == 'l':
                                    total_triple_n[4] += 1
                                elif neighNode[0] == 'm':
                                    total_triple_n[5] += 1
                    elif centerNode[0] == 'm':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd':
                                    total_triple_n[6] += 1
                                elif neighNode[0] == 'l':
                                    total_triple_n[7] += 1
                                elif neighNode[0] == 'm':
                                    total_triple_n[8] += 1
        het_walk_f.close()

        for i in range(len(total_triple_n)):
            total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
        print("sampling ratio computing finish.")
        # print(total_triple_n)
        return total_triple_n

    def sample_het_walk_triple(self):
        print("sampling triple relations ...")
        triple_list = [[] for k in range(9)]
        window = self.args.window
        walk_L = self.args.walk_L
        D_n = self.args.D_n
        L_n = self.args.L_n
        M_n = self.args.M_n
        triple_sample_p = self.triple_sample_p  # use sampling to avoid memory explosion

        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")

        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'd':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd' and random.random() < triple_sample_p[0]:
                                    negNode = random.randint(0, D_n - 1)
                                    while len(self.d_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, D_n - 1)
                                    # random negative sampling get similar performance as noise distribution sampling
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[0].append(triple)
                                elif neighNode[0] == 'l' and random.random() < triple_sample_p[1]:
                                    negNode = random.randint(0, L_n - 1)
                                    while len(self.l_d_list_total[negNode]) == 0:
                                        negNode = random.randint(0, L_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[1].append(triple)
                                elif neighNode[0] == 'm' and random.random() < triple_sample_p[2]:
                                    negNode = random.randint(0, M_n - 1)
                                    while len(self.m_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, M_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[2].append(triple)
                    elif centerNode[0] == 'l':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd' and random.random() < triple_sample_p[3]:
                                    negNode = random.randint(0, D_n - 1)
                                    while len(self.d_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, D_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[3].append(triple)
                                elif neighNode[0] == 'l' and random.random() < triple_sample_p[4]:
                                    negNode = random.randint(0, L_n - 1)
                                    while len(self.l_d_list_total[negNode]) == 0:
                                        negNode = random.randint(0, L_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[4].append(triple)
                                elif neighNode[0] == 'm' and random.random() < triple_sample_p[5]:
                                    negNode = random.randint(0, M_n - 1)
                                    while len(self.m_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, M_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[5].append(triple)
                    elif centerNode[0] == 'm':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'd' and random.random() < triple_sample_p[6]:
                                    negNode = random.randint(0, D_n - 1)
                                    while len(self.d_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, D_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[6].append(triple)
                                elif neighNode[0] == 'l' and random.random() < triple_sample_p[7]:
                                    negNode = random.randint(0, L_n - 1)
                                    while len(self.l_d_list_total[negNode]) == 0:
                                        negNode = random.randint(0, L_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[7].append(triple)
                                elif neighNode[0] == 'm' and random.random() < triple_sample_p[8]:
                                    negNode = random.randint(0, M_n - 1)
                                    while len(self.m_l_list_total[negNode]) == 0:
                                        negNode = random.randint(0, M_n - 1)
                                    triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple_list[8].append(triple)
        het_walk_f.close()
        return triple_list
